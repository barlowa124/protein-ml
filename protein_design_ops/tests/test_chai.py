"""Chai-1 subprocess driver: fake `chai` binary, no weights needed."""
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from design_ops.chai import _cif_bfactor_plddt, _npz_scores, fold_one, fold_records


def _fake_chai(tmp_path, script_body):
    """Write an executable shell script standing in for `chai fold`."""
    exe = tmp_path / "chai"
    exe.write_text(f"#!/bin/sh\n{script_body}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def _writer_script(tmp_path):
    """chai stand-in: writes scores.model_idx_0.npz + pred.model_idx_0.cif."""
    helper = tmp_path / "fake_chai.py"
    helper.write_text(
        "import sys, numpy as np\n"
        "out = sys.argv[1]\n"
        "np.savez(out + '/scores.model_idx_0.npz', aggregate_score=0.71,\n"
        "         ptm=0.62, iptm=0.55, has_clashes=0,\n"
        "         plddt=np.array([80.0, 85.0, 75.0]))\n"
        "open(out + '/pred.model_idx_0.cif', 'w').write('data_x\\n')\n"
    )
    # chai fold <fasta> <outdir>: argv[1]=fold argv[2]=fasta argv[3]=outdir
    return f'"{sys.executable}" "{helper}" "$3"'


class TestNpzScores:
    def test_scalars_and_plddt(self, tmp_path):
        p = tmp_path / "s.npz"
        np.savez(p, aggregate_score=0.71, ptm=0.62, iptm=0.55,
                 has_clashes=1, plddt=np.array([80.0, 60.0]))
        s = _npz_scores(p)
        assert s["aggregate_score"] == 0.71
        assert s["ptm"] == 0.62
        assert s["iptm"] == 0.55
        assert s["has_clashes"] == 1
        assert s["plddt_mean"] == 70.0
        assert s["plddt_min"] == 60.0

    def test_missing_keys_omitted(self, tmp_path):
        p = tmp_path / "s.npz"
        np.savez(p, ptm=0.5)
        s = _npz_scores(p)
        assert "aggregate_score" not in s
        assert "plddt_mean" not in s
        assert s["ptm"] == 0.5


class TestCifBfactorFallback:
    def _cif(self):
        return (
            "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n"
            "_atom_site.auth_atom_id\n_atom_site.B_iso_or_equiv\n"
            "ATOM 1 CA 80.0\nATOM 2 CA 60.0\nATOM 3 N 50.0\n#\n"
        )

    def test_ca_bfactor_mean_min(self, tmp_path):
        p = tmp_path / "x.cif"
        p.write_text(self._cif())
        mean, mn = _cif_bfactor_plddt(p)
        assert mean == 70.0
        assert mn == 60.0

    def test_no_atom_site_returns_none(self, tmp_path):
        p = tmp_path / "x.cif"
        p.write_text("data_x\n")
        assert _cif_bfactor_plddt(p) == (None, None)


class TestFoldOne:
    def test_happy_path(self, tmp_path):
        chai = _fake_chai(tmp_path, _writer_script(tmp_path))
        wd = tmp_path / "work"
        wd.mkdir()
        m = fold_one("ACDEFGHIK", chai, wd)
        assert m["aggregate_score"] == 0.71
        assert m["plddt_mean"] == 80.0
        assert m["structure_path"].endswith("pred.model_idx_0.cif")

    def test_extra_args_forwarded(self, tmp_path):
        # assert args land on the command line: fake records argv to file
        marker = tmp_path / "argv.txt"
        chai = _fake_chai(
            tmp_path,
            f'echo "$@" > "{marker}"\n' + _writer_script(tmp_path),
        )
        wd = tmp_path / "work"
        wd.mkdir()
        fold_one("ACDE", chai, wd, ["--seed", "7"])
        assert "--seed 7" in marker.read_text()

    def test_fasta_header_has_entity_type(self, tmp_path):
        # chai requires >protein|name; a bare name fails server-side with
        # "native is not a valid entity type" -- pin the prefix contract
        chai = _fake_chai(tmp_path, 'cp "$2" "' + str(tmp_path)
                          + '/got.fasta"\n' + _writer_script(tmp_path))
        wd = tmp_path / "work"
        wd.mkdir()
        fold_one("ACDE", chai, wd)
        assert (tmp_path / "got.fasta").read_text().startswith(
            ">protein|")

    def test_nonzero_exit_raises(self, tmp_path):
        chai = _fake_chai(tmp_path, "echo 'boom: no weights' >&2; exit 3")
        wd = tmp_path / "work"
        wd.mkdir()
        with pytest.raises(RuntimeError, match="exited 3.*no weights"):
            fold_one("ACDE", chai, wd)

    def test_missing_outputs_raises(self, tmp_path):
        chai = _fake_chai(tmp_path, "exit 0")  # exits 0 but writes nothing
        wd = tmp_path / "work"
        wd.mkdir()
        with pytest.raises(RuntimeError, match="no scores/model files"):
            fold_one("ACDE", chai, wd)


class TestFoldRecords:
    def _records(self):
        return [
            {"seq": "ACDEFGHIK", "is_native": True, "header": "n"},
            {"seq": "ACDYYHIKL", "is_native": False, "header": "d1"},
        ]

    def test_records_get_fold_and_structure(self, tmp_path):
        chai = _fake_chai(tmp_path, _writer_script(tmp_path))
        folds_dir = tmp_path / "folds"
        out = fold_records(
            self._records(), {"chai": {"command": chai}}, folds_dir)
        assert all(r["fold_backend"] == "chai1" for r in out)
        assert all(r["fold"]["aggregate_score"] == 0.71 for r in out)
        cifs = sorted(folds_dir.glob("*.cif"))
        assert len(cifs) == 2
        assert all(Path(r["structure_path"]).exists() for r in out)

    def test_command_missing_clear_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            fold_records(
                self._records(),
                {"chai": {"command": "chai-definitely-missing-xyz"}},
                tmp_path / "folds",
            )

    def test_empty_seq_skipped(self, tmp_path):
        chai = _fake_chai(tmp_path, _writer_script(tmp_path))
        recs = self._records() + [{"seq": "", "is_native": False}]
        out = fold_records(recs, {"chai": {"command": chai}},
                           tmp_path / "folds")
        assert out[-1]["fold"] is None
        assert out[-1]["fold_error"] == "empty seq"

    def test_failed_call_recorded_not_raised(self, tmp_path):
        chai = _fake_chai(tmp_path, "echo died >&2; exit 1")
        out = fold_records(self._records(), {"chai": {"command": chai}},
                           tmp_path / "folds")
        assert all(r["fold"] is None for r in out)
        assert all("exited 1" in r["fold_error"] for r in out)
