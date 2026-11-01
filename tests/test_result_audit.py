"""Check that plausible but corrupted aggregate numbers cannot pass audit."""
import copy
from dataclasses import asdict
import unittest
import torch
from audit_results import audit_policy
from evaluate import metrics
from src.models import ExitPolicy


class ResultAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.z=[torch.tensor([[5.,0.],[0.,5.],[1.,0.],[0.,1.]])]*2
        self.z.append(torch.tensor([[0.,5.],[0.,5.],[5.,0.],[5.,0.]]))
        self.y=torch.tensor([0,1,0,0]);self.costs=dict(exit_macs=[5,7,10],backbone_macs=10)
        self.policy=ExitPolicy((.9,1.1))
        self.report=metrics(self.z,self.y,self.policy,self.costs)

    def test_reconstruct_mixed_exit_population(self):
        a=audit_policy(self.z,self.y,asdict(self.policy),self.costs,self.report)
        self.assertEqual(a['exit_counts'],[2,0,2]);self.assertEqual(a['correct'],4)
        self.assertEqual(a['mean_macs'],7.5)

    def test_reject_corrupted_accuracy_cost_or_exit_distribution(self):
        for key,value in [('accuracy',.9),('mean_macs',7.4),('exit_counts',[1,1,2]),('beneficial_early_exit_fraction',0.)]:
            with self.subTest(key=key):
                altered=copy.deepcopy(self.report);altered[key]=value
                with self.assertRaises(RuntimeError):
                    audit_policy(self.z,self.y,asdict(self.policy),self.costs,altered)

    def test_full_depth_null_early_error(self):
        p=ExitPolicy();r=metrics(self.z,self.y,p,self.costs)
        a=audit_policy(self.z,self.y,asdict(p),self.costs,r)
        self.assertIsNone(a['early_exit_error_rate'])


if __name__=='__main__':unittest.main()
