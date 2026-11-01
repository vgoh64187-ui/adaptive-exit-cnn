"""Check budget routing accounting against explicit sample outcomes."""
import unittest
import torch
from evaluate import choose_policies, metrics
from src.models import ExitPolicy


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.outputs=[torch.tensor([[3.,0.],[0.,3.],[3.,0.],[0.,3.]]),
                      torch.tensor([[2.,0.],[2.,0.],[0.,2.],[0.,2.]]),
                      torch.tensor([[4.,0.],[4.,0.],[0.,4.],[0.,4.]])]
        self.labels=torch.zeros(4,dtype=torch.long)
        self.costs=dict(exit_macs=[5.,7.,10.],backbone_macs=10.)

    def test_error_transfer_identity(self):
        result=metrics(self.outputs,self.labels,ExitPolicy((0.,0.)),self.costs)
        self.assertEqual(result['exit_counts'],[4,0,0])
        self.assertEqual(result['cost_ratio'],.5)
        self.assertEqual(result['harmful_early_exit_fraction'],.25)
        self.assertEqual(result['beneficial_early_exit_fraction'],.25)
        final=float(self.outputs[-1].argmax(1).eq(self.labels).float().mean())
        self.assertAlmostEqual(result['accuracy']-final,
            result['beneficial_early_exit_fraction']-result['harmful_early_exit_fraction'])

    def test_fixed_controls_and_selected_budgets(self):
        records=choose_policies(self.outputs,self.labels,self.costs,[1.,1.,1.])
        fixed={r['criterion']:r for r in records if r['criterion'].startswith('fixed') or r['criterion']=='full_depth'}
        for name,counts,cost in [('fixed_exit_1',[4,0,0],.5),('fixed_exit_2',[0,4,0],.7),('full_depth',[0,0,4],1.)]:
            self.assertEqual(fixed[name]['selection']['exit_counts'],counts)
            self.assertEqual(fixed[name]['selection']['cost_ratio'],cost)
        for record in records:
            if record['criterion']=='budget' and 'policy' in record:
                self.assertLessEqual(record['selection']['cost_ratio'],record['budget'])


if __name__=='__main__':unittest.main()
