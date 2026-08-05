import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import lc0_experiment_driver as driver  # noqa: E402


class LC0ExperimentDriverContracts(unittest.TestCase):
    def test_arms_use_full_value_v19_b_foundation(self):
        self.assertEqual(driver.RAW, "data/raw/combined_v19_B")
        self.assertEqual(
            driver.PROCESSED,
            "data/processed/combined_v19_B_r50h60_aux",
        )
        self.assertEqual(
            driver.REFERENCE_PROCESSED,
            "data/processed/combined_v19_B_r50h60",
        )
        self.assertEqual(
            driver.PROCESSED_EXACT,
            "data/processed/combined_v19_B_r50h60_aux_exact",
        )
        self.assertEqual(
            driver.CONTROL,
            "models/candidates/v19_B/best_value_net.pt",
        )
        self.assertEqual(driver.MODEL_PREFIX, "lc0b_exact")

    def test_only_legal_mask_uses_cleaned_policy_weights(self):
        self.assertEqual(driver.ARM_DATA["legal_mask"], driver.PROCESSED)
        self.assertEqual(
            driver.ARM_DATA["moves_left"], driver.PROCESSED_EXACT)
        self.assertEqual(
            driver.ARM_DATA["ema"], driver.REFERENCE_PROCESSED)

    def test_arms_change_one_feature_each(self):
        self.assertEqual(driver.ARMS["control"], [])
        self.assertEqual(driver.ARMS["legal_mask"], ["--legal-policy-mask"])
        self.assertEqual(driver.ARMS["attention"],
                         ["--policy-head", "attention"])
        self.assertEqual(driver.ARMS["ema"], ["--ema-decay", "0.999"])
        self.assertNotIn("--legal-policy-mask", driver.ARMS["moves_left"])

    def test_auxiliary_corpus_contract_requires_both_new_arrays(self):
        self.assertIn("moves_left.npy", driver.REQUIRED_PROCESSED)
        self.assertIn("moves_left_weights.npy", driver.REQUIRED_PROCESSED)
        self.assertIn("legal_masks_packed.npy", driver.REQUIRED_PROCESSED)


if __name__ == "__main__":
    unittest.main()
