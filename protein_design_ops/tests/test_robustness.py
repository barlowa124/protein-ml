"""Robustness battery: FASTA parsing traps, backbone chain edges,
and report-metric degenerate inputs."""

import numpy as np
import pytest

from design_ops.backbone import parse_chain
from design_ops.generate import parse_mpnn_fasta
from design_ops.report import identity, pairwise_identities, spearman


def _fa(text, tmp_path):
    p = tmp_path / "x.fa"
    p.write_text(text)
    return str(p)


class TestMpnnFasta:
    H = ">1abc, score=0.5, seq_recovery=0.6, designed_chains=[A]"
    S = ">1abc, sample=1, T=0.1, seed=3, score=0.4, seq_recovery=0.5, designed_chains=[A]"

    def test_wrapped_sequence_accumulates(self, tmp_path):
        # multi-line FASTA must concatenate, not truncate to line 1
        path = _fa(f"{self.H}\nMKTA\nACDE\n{self.S}\nWXYZ\n", tmp_path)
        recs = parse_mpnn_fasta(path)
        assert recs[0]["seq"] == "MKTAACDE"
        assert recs[1]["seq"] == "WXYZ"

    def test_header_only_marker_lines_skipped(self, tmp_path):
        # MPNN emits >T=0.1 block separators with no sequence
        path = _fa(f"{self.H}\nMKTA\n>T=0.1\n{self.S}\nWXYZ\n", tmp_path)
        recs = parse_mpnn_fasta(path)
        assert len(recs) == 2 and recs[1]["seq"] == "WXYZ"

    def test_seq_without_header_ignored(self, tmp_path):
        path = _fa("ORPHANSEQ\n" + self.H + "\nMKTA\n", tmp_path)
        recs = parse_mpnn_fasta(path)
        assert len(recs) == 1 and recs[0]["seq"] == "MKTA"

    def test_header_with_seq_but_no_score_raises(self, tmp_path):
        path = _fa(">x, designed_chains=[A]\nMKTA\n", tmp_path)
        with pytest.raises(ValueError, match="score"):
            parse_mpnn_fasta(path)

    def test_empty_file(self, tmp_path):
        assert parse_mpnn_fasta(_fa("", tmp_path)) == []

    def test_native_flag_semantics(self, tmp_path):
        path = _fa(f"{self.H}\nMKTA\n{self.S}\nWXYZ\n", tmp_path)
        recs = parse_mpnn_fasta(path)
        assert recs[0]["is_native"] is True
        assert recs[1]["is_native"] is False


class TestBackboneEdges:
    def _pdb_line(self, resseq, resname="ALA", chain="A", atom="CA"):
        return (f"ATOM  {1:>5} {atom:>4} {resname:>3} {chain}"
                f"{resseq:>4}    {1.0:8.3f}{2.0:8.3f}{3.0:8.3f}"
                f"{1.00:6.2f}{20.0:6.2f}           {atom[0]:>2}")

    def test_wrong_chain_raises(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text(self._pdb_line(1) + "\n")
        with pytest.raises(ValueError, match="chain"):
            parse_chain(str(p), "B")

    def test_nonstandard_residue_skipped(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text(self._pdb_line(1) + "\n"
                     + self._pdb_line(2, resname="MSE") + "\n"
                     + self._pdb_line(3, resname="GLY") + "\n")
        out = parse_chain(str(p), "A")
        assert out["native_seq"] == "AG"
        assert out["resseqs"] == [1, 3]

    def test_short_lines_skipped_not_crash(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text("ATOM  short\n" + self._pdb_line(1) + "\n")
        assert parse_chain(str(p), "A")["resseqs"] == [1]

    def test_endmdl_limits_to_model1(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text(self._pdb_line(1) + "\nENDMDL\n"
                     + self._pdb_line(2) + "\n")
        assert parse_chain(str(p), "A")["resseqs"] == [1]


class TestReportMetricEdges:
    def test_identity_bounds_and_mismatch(self):
        assert identity("AAAA", "AAAA") == 1.0
        assert identity("AAAA", "WWWW") == 0.0
        with pytest.raises(AssertionError):
            identity("AAA", "AAAA")

    def test_pairwise_count(self):
        assert len(pairwise_identities(["AAA", "AAB", "ABB"])) == 3

    def test_spearman_constant_input_nan(self):
        v = spearman([1.0, 1.0, 1.0], [0.1, 0.5, 0.9])
        assert np.isnan(v)

    def test_spearman_perfect(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
