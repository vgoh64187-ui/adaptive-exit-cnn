"""Prevent statistically invalid aggregation and accidental stale-policy use."""
import json
from pathlib import Path
import tempfile
import unittest
from inference import load_policy
from summarise import aggregate


def row(seed):
    return dict(head='gap',kd=0.,criterion='budget',budget=.75,allowed_drop=None,
                consensus=False,seed=seed,accuracy=.70+seed*.0001,cost_ratio=.74,
                mean_macs=400000000.,final_exit_accuracy=.75,
                exit1_accuracy=.68,exit2_accuracy=.72,
                harmful_early_exit_fraction=.03,beneficial_early_exit_fraction=.01,
                exit1_fraction=.3,exit2_fraction=.3,exit3_fraction=.4)


class ReportingTests(unittest.TestCase):
    def test_three_seed_sample_standard_deviation(self):
        rows=[row(s) for s in [17,42,73]]
        result=aggregate(rows)[0]
        self.assertEqual(result['n_seeds'],3)
        self.assertGreater(result['accuracy_std'],0)

    def test_reject_single_or_duplicate_seed(self):
        with self.assertRaises(RuntimeError):
            aggregate([row(42)])
        with self.assertRaises(RuntimeError):
            aggregate([row(42),row(42),row(17)])

    def test_policy_cannot_silently_use_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'policies.json').write_text(json.dumps({'weights_sha256':'original'}))
            with self.assertRaises(RuntimeError):
                load_policy(folder,'changed')

    def test_fixed_depth_control_does_not_override_requested_budget(self):
        from dataclasses import asdict
        from src.models import ExitPolicy
        with tempfile.TemporaryDirectory() as folder:
            controls=[dict(criterion='full_depth',policy=asdict(ExitPolicy())),
                      dict(criterion='budget',budget=.75,
                           policy=asdict(ExitPolicy(thresholds=(.8,.7))))]
            Path(folder,'policies.json').write_text(json.dumps(dict(weights_sha256='same',policies=controls)))
            policy,selected=load_policy(folder,'same',budget=.75)
            self.assertEqual(selected['criterion'],'budget')
            self.assertEqual(tuple(policy.thresholds),(.8,.7))


if __name__=='__main__':
    unittest.main()
