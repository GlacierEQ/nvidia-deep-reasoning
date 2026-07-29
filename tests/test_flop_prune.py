"""Test suite for NVIDIA Deep Reasoning FLOP Pruning Kernel."""
import unittest

class FlopPruneKernelSim:
    def prune_scores(self, scores: list, threshold: float) -> list:
        return [s if s >= threshold else 0.0 for s in scores]

class TestFlopPruneKernel(unittest.TestCase):
    def test_attention_pruning(self):
        k = FlopPruneKernelSim()
        res = k.prune_scores([0.85, 0.02, 0.44, 0.01], 0.1)
        self.assertEqual(res, [0.85, 0.0, 0.44, 0.0])

if __name__ == "__main__":
    unittest.main()
