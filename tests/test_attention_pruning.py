from __future__ import annotations

import math
import unittest

from src.attention_pruning import prune_attention_scores


class AttentionPruningTests(unittest.TestCase):
    def test_thresholding_is_deterministic_and_boundary_inclusive(self) -> None:
        result = prune_attention_scores([0.85, 0.02, 0.10, 0.44, 0.01], 0.10)
        self.assertEqual(result.output_scores, (0.85, 0.0, 0.10, 0.44, 0.0))
        self.assertEqual(result.input_count, 5)
        self.assertEqual(result.kept_count, 3)
        self.assertEqual(result.pruned_count, 2)
        self.assertAlmostEqual(result.masked_fraction, 0.4)
        self.assertFalse(result.operational_authority)

    def test_input_is_not_mutated(self) -> None:
        scores = [0.3, 0.1, 0.8]
        before = list(scores)
        prune_attention_scores(scores, 0.2)
        self.assertEqual(scores, before)

    def test_empty_input_is_valid(self) -> None:
        result = prune_attention_scores([], 0.25)
        self.assertEqual(result.output_scores, ())
        self.assertEqual(result.masked_fraction, 0.0)

    def test_rejects_non_finite_threshold(self) -> None:
        for threshold in (math.nan, math.inf, -math.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    prune_attention_scores([0.1], threshold)

    def test_rejects_non_finite_scores(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    prune_attention_scores([0.1, value], 0.2)

    def test_rejects_boolean_as_numeric_input(self) -> None:
        with self.assertRaises(ValueError):
            prune_attention_scores([True], 0.2)
        with self.assertRaises(ValueError):
            prune_attention_scores([0.2], False)

    def test_result_dictionary_preserves_non_authority_boundary(self) -> None:
        payload = prune_attention_scores([0.2, 0.8], 0.5).as_dict()
        self.assertEqual(payload["output_scores"], [0.0, 0.8])
        self.assertIs(payload["operational_authority"], False)


if __name__ == "__main__":
    unittest.main()
