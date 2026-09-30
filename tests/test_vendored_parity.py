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


# Cross-repo pin: mol-ml/dti_fusion vendors the same file and pins this
# digest. Editing any copy on either side trips the pin — update all
# copies and the constant together.
ESM_CACHE_SHA256 = \
    "2c80f1d43fffe47c126ce70e0f7342ced3458a3d902105c6275cacc338295531"

# Same arrangement for the canonical conformal helpers: mol-ml/
# comp_tox_pipeline vendors it as eval/conformal_shared.py.
CONFORMAL = ROOT / "protein_stability_uncertainty" / "src" / "protstab" / \
    "conformal.py"
CONFORMAL_SHA256 = \
    "fcba54721a3f106e3864ab43ad0cb61caf273c41426d7bfa649b542f5f72af00"


class VendoredParityTests(unittest.TestCase):
    def test_esm_cache_copies_identical(self):
        texts = [p.read_bytes() for p in VENDORED]
        for p, t in zip(VENDORED[1:], texts[1:]):
            self.assertEqual(
                texts[0], t,
                f"{p} drifted from {VENDORED[0]} — update both copies "
                "together or this test is failing for a real reason")

    def test_esm_cache_cross_repo_pin(self):
        import hashlib
        for p in VENDORED:
            self.assertEqual(
                hashlib.sha256(p.read_bytes()).hexdigest(),
                ESM_CACHE_SHA256,
                f"{p} no longer matches the shared vendored digest — "
                "mol-ml/dti_fusion carries the same file; sync both repos "
                "and update the pin together")

    def test_conformal_cross_repo_pin(self):
        import hashlib
        self.assertEqual(
            hashlib.sha256(CONFORMAL.read_bytes()).hexdigest(),
            CONFORMAL_SHA256,
            "protstab/conformal.py no longer matches the shared digest — "
            "mol-ml/comp_tox_pipeline vendors the same helpers; sync both "
            "repos and update the pin together")


if __name__ == "__main__":
    unittest.main()
