"""Behavioral tests: routing causality, active-batch compaction, and gradients."""
import unittest
import torch
from src.models import ResNet18, ExitPolicy, route_cached, profile_macs


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.x = torch.randn(7, 3, 32, 32)

    def test_dynamic_matches_cached_mixed_routes(self):
        for head in ['gap', 'local_global', 'mlp']:
            model = ResNet18(classes=5, head=head, width=8).eval()
            with torch.no_grad():
                outputs = model(self.x)
                values = outputs[0].softmax(1).max(1).values
                threshold = values.median().item()
                for consensus in [False, True]:
                    policy = ExitPolicy((threshold, .0), consensus=consensus)
                    expected, ids = route_cached(outputs, policy)
                    actual, actual_ids = model.predict_dynamic(self.x, policy)
                    self.assertGreater(len(ids.unique()), 1)
                    torch.testing.assert_close(actual_ids, ids)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_early_exit_skips_deep_layers(self):
        model = ResNet18(classes=5, head='local_global', width=8).eval()
        visited = []
        handles = [stage.register_forward_hook(lambda m, a, b, i=i: visited.append(i))
                   for i, stage in enumerate(model.stages)]
        model.predict_dynamic(self.x, ExitPolicy((0., 0.)))
        self.assertEqual(visited, [0, 1])
        for h in handles:
            h.remove()

    def test_disabled_exit_matches_final(self):
        model = ResNet18(classes=5, head='local_global', width=8).eval()
        with torch.no_grad():
            full = model(self.x)[-1]
            actual, ids = model.predict_dynamic(self.x, ExitPolicy())
        torch.testing.assert_close(actual, full)
        self.assertTrue(ids.eq(2).all())

    def test_training_gradients_and_cost_order(self):
        model = ResNet18(classes=5, head='local_global', width=8)
        labels = torch.arange(len(self.x)) % 5
        loss = sum(torch.nn.functional.cross_entropy(z, labels) for z in model(self.x))
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        costs = profile_macs(model)
        self.assertTrue(costs['exit_macs'][0] < costs['exit_macs'][1] < costs['exit_macs'][2])
        self.assertGreater(costs['exit_macs'][2], costs['backbone_macs'])

    def test_baseline_has_one_output(self):
        model = ResNet18(classes=5, width=8)
        self.assertEqual(len(model(self.x)), 1)


if __name__ == '__main__':
    unittest.main()
