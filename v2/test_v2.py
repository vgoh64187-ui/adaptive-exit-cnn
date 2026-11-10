"""Targeted checks for the V2 residual head and routing computation."""
import unittest
import numpy as np
import torch
from v2.routing_validation import evaluate, select
from v2.src.models import ExitPolicy, ResNet18, ResidualLocalGlobalHead, route_cached

torch.set_num_threads(1)


class V2RegressionTests(unittest.TestCase):
    def test_residual_head_starts_as_gap_and_learns(self):
        torch.manual_seed(7)
        head = ResidualLocalGlobalHead(128, 100)
        inputs = torch.rand(4, 128, 16, 16)
        head.eval()
        torch.testing.assert_close(head(inputs), head.fc(inputs.mean((2, 3))))
        head.train()
        head(inputs).square().mean().backward()
        self.assertGreater(head.gamma.grad.abs().item(), 0)

    def test_dynamic_path_matches_cached_outputs(self):
        torch.manual_seed(9)
        model = ResNet18(head="residual_local_global").eval()
        inputs = torch.rand(7, 3, 32, 32)
        for policy in (
            ExitPolicy(thresholds=(0., 0.)),
            ExitPolicy(thresholds=(1.1, 0.)),
            ExitPolicy(thresholds=(1.1, 1.1)),
            ExitPolicy(thresholds=(.011, .011), consensus=True),
        ):
            with torch.no_grad():
                dynamic_logits, dynamic_ids = model.predict_dynamic(inputs, policy)
                cached_logits, cached_ids = route_cached(model(inputs), policy)
            torch.testing.assert_close(dynamic_ids, cached_ids)
            torch.testing.assert_close(dynamic_logits, cached_logits, atol=1e-5, rtol=1e-5)

    def test_grid_search_matches_brute_force(self):
        rng = np.random.default_rng(42)
        size = 73
        scores = tuple(rng.random(size) for _ in range(4))
        predictions = [rng.integers(0, 3, size) for _ in range(3)]
        labels = rng.integers(0, 3, size)
        costs = {"exit_macs": [50, 75, 100], "backbone_macs": 100}
        indices = np.arange(size)
        for margin in (0., .15):
            thresholds = select(scores, predictions, labels, costs, 7, margin, .65, indices)
            chosen = evaluate(scores, predictions, labels, costs, thresholds, margin, indices)
            candidates = []
            grids = [np.unique(np.r_[0., 1.1, np.quantile(scores[i], np.linspace(0, 1, 7))])
                     for i in (0, 2)]
            for first in grids[0]:
                for second in grids[1]:
                    result = evaluate(scores, predictions, labels, costs, (first, second), margin, indices)
                    if result["cost_ratio"] <= .65:
                        candidates.append(result)
            expected = max(candidates, key=lambda result: (result["correct"], -result["cost_ratio"]))
            self.assertEqual(chosen["correct"], expected["correct"])
            self.assertAlmostEqual(chosen["cost_ratio"], expected["cost_ratio"])


if __name__ == "__main__":
    unittest.main()
