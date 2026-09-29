import unittest

from design_ops.report import identity, pairwise_identities, spearman


class ReportTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(identity("ACGT", "ACGT"), 1.0)
        self.assertEqual(identity("ACGT", "AGGT"), 0.75)
        self.assertEqual(identity("AAAA", "TTTT"), 0.0)

    def test_pairwise(self):
        ids = pairwise_identities(["AAAA", "AATT", "AATT"])
        self.assertEqual(len(ids), 3)
        self.assertAlmostEqual(ids[0], 0.5)
        self.assertAlmostEqual(ids[2], 1.0)

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [1, 2, 3]), 1.0)

    def _records(self, n=3):
        recs = [{"seq": "AAAA", "mpnn_score": 1.0, "seq_recovery": 0.5,
                 "is_native": True, "esm_pll": -2.0}]
        for i in range(n):
            recs.append({"seq": f"AAL{'ACD'[i]}".ljust(4, 'A'),
                         "mpnn_score": 1.0 + i * 0.1, "seq_recovery": 0.3,
                         "is_native": False, "esm_pll": -2.0 + i * 0.1})
        return recs

    def test_top_k_below_one_raises(self):
        import os
        import tempfile

        from design_ops.report import report

        bb = {"pdb": "x.pdb", "chain": "A", "n_residues": 4}
        cfg = {"report": {"top_k": 0},
               "backbone": {"id": "t"}, "mpnn": {"repo_path": ".",
               "model_name": "m", "weights": "w", "num_seq_per_target": 3,
               "sampling_temp": 0.1, "seed": 1},
               "esm": {"model": "m"}}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                report(self._records(), bb, cfg,
                       os.path.join(td, "r.json"), os.path.join(td, "r.png"))


if __name__ == "__main__":
    unittest.main()
