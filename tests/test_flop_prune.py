"""Compatibility proof for the repository's bounded pruning mechanism."""
from __future__ import annotations

import unittest

from src.attention_pruning import prune_attention_scores


class FlopPruneCompatibilityTests(unittest.TestCase):
    def test_threshold_mask_uses_real_reference_implementation(self) -> None:
        result = prune_attention_scores([0.85, 0.02, 0.44, 0.01], 0.1)
        self.assertEqual(result.output_scores, (0.85, 0.0, 0.44, 0.0))
        self.assertEqual(result.pruned_count, 2)
        self.assertFalse(result.operational_authority)

    def test_mask_fraction_is_not_labeled_as_measured_flop_saving(self) -> None:
        result = prune_attention_scores([0.2, 0.4, 0.8, 0.1], 0.3)
        payload = result.as_dict()
        self.assertEqual(payload["masked_fraction"], 0.5)
        self.assertNotIn("flop_reduction", payload)
        self.assertNotIn("speedup", payload)


if __name__ == "__main__":
    unittest.main()
