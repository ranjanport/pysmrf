"""Void inpainting algorithms for raster elevation models.

Cites:
    Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).
    An improved simple morphological filter for producing ground surfaces from
    LiDAR point clouds. ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def inpaint_nans_by_springs(
    A: np.ndarray,
    inplace: bool = False,
    max_iter: int = 500,
    rtol: float = 1e-4,
) -> np.ndarray:
    """Inpaint missing cells (NaNs) in a 2D grid using a 4-connected spring / Laplacian network.

    This implements Pingel's spring model where missing cells are modeled as nodes
    connected by elastic springs (unit stiffness) to their 4-cardinal neighbors,
    anchored to known elevations as fixed Dirichlet boundaries.

    Parameters
    ----------
    A : np.ndarray
        2D raster grid with NaN values to fill.
    inplace : bool
        If True, modifies A in-place. If False, returns a filled copy.
    max_iter : int
        Maximum LSQR solver iterations. Default is 500.
    rtol : float
        Relative tolerance for solver convergence. Default is 1e-4.

    Returns
    -------
    filled : np.ndarray
        2D grid with NaNs interpolated.
    """
    nanmat = np.isnan(A)
    num_nans = int(np.sum(nanmat))

    if num_nans == 0:
        return A if inplace else A.copy()

    m, n = A.shape
    total_cells = m * n

    if num_nans == total_cells:
        # All elements are NaN
        res = np.zeros_like(A)
        if inplace:
            A[:] = res
            return A
        return res

    nan_list = np.flatnonzero(nanmat)
    known_list = np.flatnonzero(~nanmat)

    # 4-cardinal neighbor offsets: right, left, up, down
    r, c = np.unravel_index(nan_list, (m, n))
    dr = np.array([0, 0, -1, 1], dtype=np.int32)
    dc = np.array([1, -1, 0, 0], dtype=np.int32)

    nr = r[:, None] + dr[None, :]
    nc = c[:, None] + dc[None, :]

    valid = (nr >= 0) & (nr < m) & (nc >= 0) & (nc < n)
    u_nodes = np.repeat(nan_list, 4)[valid.ravel()]
    v_nodes = (nr * n + nc)[valid]

    # Ensure undirected canonical edge representation (u < v)
    min_node = np.minimum(u_nodes, v_nodes)
    max_node = np.maximum(u_nodes, v_nodes)

    edge_keys = min_node.astype(np.int64) * total_cells + max_node.astype(np.int64)
    unique_edges = np.unique(edge_keys)

    u_unique = (unique_edges // total_cells).astype(np.int32)
    v_unique = (unique_edges % total_cells).astype(np.int32)
    n_edges = len(unique_edges)

    row_indices = np.repeat(np.arange(n_edges, dtype=np.int32), 2)
    col_indices = np.column_stack((u_unique, v_unique)).ravel()
    data = np.tile([1.0, -1.0], n_edges)

    springs = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_edges, total_cells),
        dtype=np.float64,
    ).tocsr()

    known_vals = A.ravel()[known_list]
    rhs = -springs[:, known_list] @ known_vals

    nan_submatrix = springs[:, nan_list]

    # Solve least squares system
    sol = lsqr(nan_submatrix, rhs, iter_lim=max_iter, atol=rtol, btol=rtol)[0]

    out = A if inplace else A.copy()
    out.ravel()[nan_list] = sol

    # Guard against any isolated disconnected regions that might evaluate to NaN
    if np.any(np.isnan(out)):
        out = _fill_remaining_nans_nearest(out)

    return out


def inpaint_nans_by_fda(
    A: np.ndarray,
    fast: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Inpaint missing cells (NaNs) using finite difference Laplacian approximation.

    Parameters
    ----------
    A : np.ndarray
        2D grid with NaNs.
    fast : bool
        If True, solves only for dilated neighborhood around missing data.
    inplace : bool
        If True, writes directly into input array.

    Returns
    -------
    filled : np.ndarray
        2D grid with NaNs filled.
    """
    nanmat = np.isnan(A)
    num_nans = int(np.sum(nanmat))
    if num_nans == 0:
        return A if inplace else A.copy()

    m, n = A.shape
    nan_list = np.flatnonzero(nanmat)
    known_list = np.flatnonzero(~nanmat)

    index = np.arange(m * n, dtype=np.int64).reshape((m, n))

    i = np.hstack(
        (
            np.tile(index[1:-1, :].ravel(), 3),
            np.tile(index[:, 1:-1].ravel(), 3),
        )
    )
    j = np.hstack(
        (
            index[0:-2, :].ravel(),
            index[2:, :].ravel(),
            index[1:-1, :].ravel(),
            index[:, 0:-2].ravel(),
            index[:, 2:].ravel(),
            index[:, 1:-1].ravel(),
        )
    )
    data = np.hstack(
        (
            np.ones(2 * n * (m - 2), dtype=np.float64),
            -2.0 * np.ones(n * (m - 2), dtype=np.float64),
            np.ones(2 * m * (n - 2), dtype=np.float64),
            -2.0 * np.ones(m * (n - 2), dtype=np.float64),
        )
    )

    if fast:
        from scipy.ndimage import binary_dilation

        dilated_mask = binary_dilation(nanmat)
        goodrows = np.isin(i, index[dilated_mask])
        i = i[goodrows]
        j = j[goodrows]
        data = data[goodrows]

    fda = sparse.coo_matrix((data, (i, j)), (m * n, m * n), dtype=np.float64).tocsr()

    known_vals = A.ravel()[known_list]
    rhs = -fda[:, known_list] @ known_vals

    nan_unique = np.unique(nan_list)
    k = fda[:, nan_unique].nonzero()[0]
    a = fda[k][:, nan_list]

    sol = lsqr(a, rhs[k])[0]

    out = A if inplace else A.copy()
    out.ravel()[nan_list] = sol

    if np.any(np.isnan(out)):
        out = _fill_remaining_nans_nearest(out)

    return out


def _fill_remaining_nans_nearest(A: np.ndarray) -> np.ndarray:
    """Fill any remaining NaNs in A using nearest valid neighbor via Euclidean distance transform."""
    from scipy.ndimage import distance_transform_edt

    nan_mask = np.isnan(A)
    if not np.any(nan_mask):
        return A
    if np.all(nan_mask):
        return np.zeros_like(A)

    indices = distance_transform_edt(
        nan_mask, return_distances=False, return_indices=True
    )
    filled = A.copy()
    filled[nan_mask] = A[tuple(ind[nan_mask] for ind in indices)]
    return filled


def inpaint_dem(
    A: np.ndarray,
    method: str = "spring",
    inplace: bool = False,
) -> np.ndarray:
    """Unified DEM void inpainting interface.

    Parameters
    ----------
    A : np.ndarray
        2D elevation grid with NaNs.
    method : str
        Inpainting algorithm: 'spring' (default, Pingel 2013), 'fda',
        'nearest', or 'idw'.
    inplace : bool
        If True, writes directly into input array.

    Returns
    -------
    filled : np.ndarray
        Inpainted 2D grid.
    """
    method_lower = method.lower()
    if method_lower == "spring":
        return inpaint_nans_by_springs(A, inplace=inplace)
    elif method_lower == "fda":
        return inpaint_nans_by_fda(A, inplace=inplace)
    elif method_lower in ("nearest", "idw"):
        return _fill_remaining_nans_nearest(A if inplace else A.copy())
    else:
        raise ValueError(
            f"Unknown inpaint method: '{method}'. Choose 'spring', 'fda', or 'nearest'."
        )
