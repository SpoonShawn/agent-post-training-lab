import unittest
from training.transaction_grpo import clipped_objective
from scripts.transaction_grpo_pilot import protocol

class GRPOPilotTests(unittest.TestCase):
    def test_frozen_boundaries_and_selection(self):
        spec,cases=protocol(False)
        self.assertEqual(len(cases),24); self.assertEqual(len(set(spec['selected_group_ids'])),24)
        self.assertTrue(all(c['split']=='train' for c in cases)); self.assertEqual({c['category'] for c in cases},{'apply','rollback','permission'})
        self.assertEqual(spec['eval_splits'],['confirmation_id','confirmation_ood']); self.assertTrue(spec['probe_excluded'])
    def test_objective_has_gradient_and_kl(self):
        torch = __import__('unittest').SkipTest
        try: import torch as _torch
        except ModuleNotFoundError: self.skipTest('torch is only available on SuperPOD')
        torch = _torch
        n=torch.tensor([-.2,-.4],requires_grad=True); o=torch.tensor([-.3,-.3]); r=torch.tensor([-.25,-.35])
        v,ratio,kl=clipped_objective(n,o,r,1.); self.assertTrue(torch.isfinite(v).all()); self.assertTrue((kl>=0).all()); v.mean().backward(); self.assertIsNotNone(n.grad)

if __name__=='__main__': unittest.main()
