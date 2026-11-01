import unittest
import torch
from diagnostics import reliability


class ReliabilityTests(unittest.TestCase):
    def test_perfectly_correct_high_confidence(self):
        stats=reliability(torch.tensor([[20.,-20.],[-20.,20.]]),torch.tensor([0,1]))
        self.assertAlmostEqual(stats['accuracy'],1.)
        self.assertLess(stats['ece'],1e-6)
        self.assertEqual(sum(b['n'] for b in stats['bins']),2)

    def test_temperature_keeps_top_class_and_changes_confidence(self):
        logits=torch.tensor([[4.,1.],[.5,2.],[-1.,3.]])
        labels=torch.tensor([0,0,1])
        raw=reliability(logits,labels,1.)
        scaled=reliability(logits,labels,3.)
        self.assertEqual(raw['accuracy'],scaled['accuracy'])
        self.assertNotEqual(raw['nll'],scaled['nll'])
        self.assertAlmostEqual(raw['risk_coverage'][-1]['risk'],1/3,places=6)


if __name__=='__main__':unittest.main()
