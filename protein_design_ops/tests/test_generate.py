import tempfile
import unittest

from design_ops.generate import parse_mpnn_fasta

FASTA = """\
>1L2Y, score=1.9315, global_score=1.9315, fixed_chains=[], designed_chains=['A'], model_name=v_48_020, git_hash=abc, seed=573
NLYIQWLKDGGPSSGRPPPS
>T=0.1, sample=1, score=1.1755, global_score=1.1755, seq_recovery=0.4000
EALQRWLARGGERSGQPRPV
>T=0.1, sample=2, score=1.1848, global_score=1.1848, seq_recovery=0.3500
KALEEWLALGGENAGLPRPV
"""


class GenerateTests(unittest.TestCase):
    def test_parse_fasta(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False)
        tmp.write(FASTA)
        tmp.close()
        records = parse_mpnn_fasta(tmp.name)
        self.assertEqual(len(records), 3)
        native = records[0]
        self.assertTrue(native["is_native"])
        self.assertEqual(native["seq"], "NLYIQWLKDGGPSSGRPPPS")
        self.assertAlmostEqual(native["mpnn_score"], 1.9315)
        self.assertFalse(records[1]["is_native"])
        self.assertAlmostEqual(records[1]["seq_recovery"], 0.4)


if __name__ == "__main__":
    unittest.main()
