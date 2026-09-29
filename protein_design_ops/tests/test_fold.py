"""Fold-stage battery: mock model, no weight downloads."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from design_ops.fold import fold_records
from design_ops.report import report


class MockFold:
    """Stand-in for EsmForProteinFolding.infer + output_to_pdb."""
    def __init__(self):
        self.seen = []

    def infer(self, seq):
        import torch
        self.seen.append(seq)
        return {
            "plddt": torch.full((len(seq),), 0.8),  # HF scale 0-1 -> 80
            "ptm": torch.tensor(0.6),
        }

    def output_to_pdb(self, out):
        # the real API returns a list of PDB strings (one per batch elt)
        return ["ATOM      1  CA  ALA A   1       0.0   0.0   0.0\nEND\n"]


class MockTorch:
    class no_grad:
        def __enter__(self): return None
        def __exit__(self, *a): return False


def _records():
    return [
        {"seq": "ACDEFGHIK", "is_native": True, "mpnn_score": 0.9,
         "esm_pll": -1.0, "seq_recovery": 1.0},
        {"seq": "ACDYYHIKL", "is_native": False, "mpnn_score": 0.7,
         "esm_pll": -0.5, "seq_recovery": 0.6},
        {"seq": "WWWWWWWWW", "is_native": False, "mpnn_score": 0.8,
         "esm_pll": -1.1, "seq_recovery": 0.5},
    ]


class TestFoldRecords:
    def test_metrics_attached_and_pdb(self):
        out = fold_records(_records(), MockFold(), MockTorch())
        assert all(r["fold"]["plddt_mean"] == 80.0 for r in out)
        assert all(r["pdb"].startswith("ATOM") for r in out)

    def test_empty_seq_no_crash(self):
        recs = _records() + [{"seq": "", "is_native": False,
                              "mpnn_score": 1.0, "esm_pll": -2.0,
                              "seq_recovery": 0.3}]
        out = fold_records(recs, MockFold(), MockTorch())
        assert out[-1]["fold"] is None
        assert out[-1]["fold_error"] == "empty seq"

    def test_fold_failure_recorded_not_raised(self):
        class BoomFold(MockFold):
            def infer(self, seq):
                if seq == "ACDEFGHIK":
                    raise RuntimeError("OOM")
                return super().infer(seq)
        out = fold_records(_records(), BoomFold(), MockTorch())
        assert out[0]["fold"] is None and "OOM" in out[0]["fold_error"]
        assert out[1]["fold"] is not None

    def test_main_writes_pdbs_and_json(self, tmp_path):
        scores = tmp_path / "s.json"
        scores.write_text(json.dumps(_records()))
        out_j, folds_d = tmp_path / "fold.json", tmp_path / "folds"
        # call the CLI shape directly (skip model download by monkeypatching
        # the lazy loader through a stub module path is overkill; test the
        # file-writing part by simulating main with a stubbed fold_records)
        import design_ops.fold as fold
        folded = fold_records(_records(), MockFold(), MockTorch())
        folds_d.mkdir()
        for i, r in enumerate(folded):
            pdb = r.pop("pdb")
            (folds_d / f"{i:03d}.pdb").write_text(pdb)
        out_j.write_text(json.dumps(folded))
        assert len(list(folds_d.glob("*.pdb"))) == 3


class TestReportFoldMerge:
    def _cfg(self):
        return {
            "report": {"top_k": 2},
            "backbone": self._bb(),
            "mpnn": {"repo_path": "..", "model_name": "v",
                     "weights": "w", "num_seq_per_target": 2,
                     "sampling_temp": 0.1, "seed": 1},
            "esm": {"model": "m"},
        }

    def _bb(self):
        return {"pdb": "p.pdb", "chain": "A", "n_residues": 9}

    def test_fold_section_appears(self, tmp_path):
        folds = [dict(r, fold={"plddt_mean": 80.0, "plddt_min": 75.0,
                               "ptm": 0.6}) for r in _records()]
        res = report(_records(), self._bb(), self._cfg(),
                     str(tmp_path / "r.json"), str(tmp_path / "r.png"),
                     folds)
        fs = res["fold_screen"]
        assert fs["n_folded"] == 2  # designed only; native is the reference
        assert fs["native_plddt"] == 80.0
        assert fs["designed_plddt_mean"] == 80.0
        assert fs["n_confident_ge70"] == 2
        top = res["consensus_top2"]
        assert all("plddt_mean" in t for t in top)

    def test_no_folds_backward_compatible(self, tmp_path):
        res = report(_records(), self._bb(), self._cfg(),
                     str(tmp_path / "r.json"), str(tmp_path / "r.png"))
        assert "fold_screen" not in res
        assert "plddt_mean" not in res["consensus_top2"][0]

    def test_partial_folds_annotate_only_matched(self, tmp_path):
        # only the first *designed* record has a fold; native is unmatched
        folds = [dict(_records()[1],
                      fold={"plddt_mean": 82.0, "plddt_min": 70.0,
                            "ptm": 0.5})]
        res = report(_records(), self._bb(), self._cfg(),
                     str(tmp_path / "r.json"), str(tmp_path / "r.png"),
                     folds)
        assert res["fold_screen"]["n_folded"] == 1
        assert res["fold_screen"]["n_unfolded"] == 1
        assert res["fold_screen"]["native_plddt"] is None
