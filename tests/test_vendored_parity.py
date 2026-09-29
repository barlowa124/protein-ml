"""Monorepo-level parity: vendored copies must stay byte-identical.

esm_cache.py is deliberately vendored into each subpackage (vendored, not
depended on — each package installs standalone). Inside this monorepo the
copies can drift silently; this test makes drift a failure instead.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VENDORED = [
    ROOT / "active_learning_loop" / "src" / "al_loop" / "esm_cache.py",
    ROOT / "protein_design_ops" / "src" / "design_ops" / "esm_cache.py",
]


class VendoredParityTests(unittest.TestCase):
    def test_esm_cache_copies_identical(self):
        texts = [p.read_bytes() for p in VENDORED]
        for p, t in zip(VENDORED[1:], texts[1:]):
            self.assertEqual(
                texts[0], t,
                f"{p} drifted from {VENDORED[0]} — update both copies "
                "together or this test is failing for a real reason")


if __name__ == "__main__":
    unittest.main()
