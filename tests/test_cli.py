"""Tests for the command-line interface (CLI)."""

from pathlib import Path
import tempfile
from click.testing import CliRunner
import numpy as np

from pysmrf.cli import cli


def test_cli_version():
    runner = CliRunner()
    res = runner.invoke(cli, ["--version"])
    assert res.exit_code == 0
    assert "PySMRF" in res.output


def test_cli_cite():
    runner = CliRunner()
    res = runner.invoke(cli, ["cite"])
    assert res.exit_code == 0
    assert "Pingel" in res.output
    assert "@article" in res.output


def test_cli_info_and_dem():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_file = Path(tmpdir) / "points.txt"
        data = np.array([
            [1.0, 1.0, 10.0],
            [2.0, 2.0, 12.0],
            [3.0, 3.0, 11.0],
        ])
        np.savetxt(txt_file, data)

        res_info = runner.invoke(cli, ["info", str(txt_file)])
        assert res_info.exit_code == 0
        assert "Points:       3" in res_info.output

        dem_file = Path(tmpdir) / "dem.tif"
        res_dem = runner.invoke(cli, ["dem", str(txt_file), "-o", str(dem_file), "--cellsize", "1.0"])
        assert res_dem.exit_code == 0
        assert dem_file.exists()
