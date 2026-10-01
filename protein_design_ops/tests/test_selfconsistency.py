"""Self-consistency stage: Kabsch CA RMSD + structure parsers."""
import json

import numpy as np
import pytest

from design_ops.selfconsistency import (
    add_selfconsistency,
    kabsch_rmsd,
    parse_ca_coords_cif,
    parse_ca_coords_pdb,
)


def _pdb_line(i, x, y, z, chain="A", resname="ALA"):
    return (
        f"ATOM  {i:>5}  CA  {resname} {chain}{i:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C\n"
    )


def _write_pdb(path, coords, chain="A"):
    path.write_text(
        "".join(_pdb_line(i + 1, *c, chain) for i, c in enumerate(coords))
        + "END\n"
    )
    return str(path)


CIF = """data_t
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.auth_atom_id
ATOM 1 N ALA A 1 0.000 0.000 0.000 90.0 N
ATOM 2 CA ALA A 1 1.500 0.000 0.000 91.5 CA
ATOM 3 C ALA A 1 2.000 1.400 0.000 88.0 C
ATOM 4 CA GLY A 2 3.300 1.900 0.200 72.0 CA
#
"""


class TestKabsch:
    def test_identical_zero(self):
        a = np.array([[0, 0, 0], [1.5, 0, 0], [3.3, 1.9, 0.2]])
        assert kabsch_rmsd(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_rigid_transform_recovers_zero(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(20, 3))
        theta = 0.7
        rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )
        b = a @ rot + np.array([10.0, -4.0, 2.0])
        assert kabsch_rmsd(b, a) == pytest.approx(0.0, abs=1e-9)

    def test_one_moved_residue(self):
        a = np.array([[float(i), 0.0, 0.0] for i in range(10)])
        b = a.copy()
        b[5] += np.array([0.0, 4.0, 0.0])
        expected = np.sqrt(16.0 / 10)  # alignment keeps near-zero shift
        got = kabsch_rmsd(b, a)
        assert 0.0 < got < 4.0
        assert got == pytest.approx(expected, rel=0.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            kabsch_rmsd(np.zeros((5, 3)), np.zeros((6, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            kabsch_rmsd(np.zeros((0, 3)), np.zeros((0, 3)))


class TestParsers:
    def test_pdb_ca_only_first_model(self, tmp_path):
        text = (
            _pdb_line(1, 1.0, 2.0, 3.0)
            + "ATOM      2  N   ALA A   1       9.0   9.0   9.0\n"
            + _pdb_line(2, 4.0, 5.0, 6.0)
            + "ENDMDL\n"
            + _pdb_line(1, 7.0, 8.0, 9.0)
        )
        coords = parse_ca_coords_pdb(text)
        assert coords.shape == (2, 3)
        assert coords[1].tolist() == [4.0, 5.0, 6.0]

    def test_pdb_other_chain_excluded_by_convention(self, tmp_path):
        # PDB parser takes all CAs; chain filtering is the caller's job
        # (backbone.py does it). Two chains -> 2 CAs.
        text = _pdb_line(1, 0.0, 0.0, 0.0, "A") + _pdb_line(1, 1.0, 1.0, 1.0, "B")
        assert parse_ca_coords_pdb(text).shape == (2, 3)

    def test_cif_header_driven(self):
        coords = parse_ca_coords_cif(CIF)
        assert coords.shape == (2, 3)
        assert coords[0].tolist() == [1.5, 0.0, 0.0]
        assert coords[1].tolist() == [3.3, 1.9, 0.2]


class TestAddSelfConsistency:
    def _backbone(self):
        return {"ca_coords": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                              [3.3, 1.9, 0.2]]}

    def test_rmsd_attached(self, tmp_path):
        pdb = _write_pdb(tmp_path / "d.pdb",
                         [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.3, 1.9, 0.2]])
        recs = [{"seq": "AAA", "structure_path": pdb,
                 "fold": {"plddt_mean": 80.0}}]
        out = add_selfconsistency(recs, self._backbone())
        assert out[0]["fold"]["bb_rmsd"] == pytest.approx(0.0, abs=1e-3)

    def test_cif_structure(self, tmp_path):
        cif = tmp_path / "pred.cif"
        cif.write_text(CIF)
        backbone = {"ca_coords": [[1.5, 0.0, 0.0], [3.3, 1.9, 0.2]]}
        recs = [{"seq": "AG", "structure_path": str(cif),
                 "fold": {"aggregate_score": 0.7}}]
        out = add_selfconsistency(recs, backbone)
        assert out[0]["fold"]["bb_rmsd"] == pytest.approx(0.0, abs=1e-3)

    def test_missing_structure_recorded(self, tmp_path):
        recs = [{"seq": "AAA", "structure_path": str(tmp_path / "nope.pdb"),
                 "fold": {"plddt_mean": 80.0}}]
        out = add_selfconsistency(recs, self._backbone())
        assert out[0]["fold"]["bb_rmsd"] is None
        assert "bb_rmsd_note" in out[0]["fold"]

    def test_no_fold_untouched(self):
        recs = [{"seq": "AAA", "fold": None, "fold_error": "OOM"}]
        out = add_selfconsistency(recs, self._backbone())
        assert "bb_rmsd" not in out[0]

    def test_length_mismatch_recorded(self, tmp_path):
        pdb = _write_pdb(tmp_path / "d.pdb", [[0.0, 0.0, 0.0]])
        recs = [{"seq": "A", "structure_path": pdb, "fold": {"ptm": 0.5}}]
        out = add_selfconsistency(recs, self._backbone())
        assert out[0]["fold"]["bb_rmsd"] is None
        assert "shape mismatch" in out[0]["fold"]["bb_rmsd_note"]


def test_backbone_emits_ca_coords():
    """parse_chain must carry ca_coords for the self-consistency stage."""
    from design_ops.backbone import parse_chain

    info = parse_chain("tests/fixtures/1L2Y.pdb", "A")
    assert len(info["ca_coords"]) == len(info["native_seq"])
    assert all(len(c) == 3 for c in info["ca_coords"])
