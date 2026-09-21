import unittest
from scripts.prepare_transaction_repaired_v1 import build

class RepairedDataTests(unittest.TestCase):
    def test_split_and_mechanism_coverage(self):
        rows=build(); self.assertEqual(len(rows),160)
        groups={s:{r['group_id'] for r in rows if r['split']==s} for s in ('train','id','ood')}
        self.assertEqual({len(v) for v in groups.values()},{12,4})
        self.assertFalse(groups['train']&groups['id']); self.assertFalse(groups['train']&groups['ood']); self.assertFalse(groups['id']&groups['ood'])
        self.assertTrue(all(r['environment']['revoke_before_commit'] for r in rows))
        self.assertEqual({r['knowledge_area'] for r in rows},{'state_revision','retry_idempotency','async_check','permission_recovery','evidence_report'})

if __name__=='__main__': unittest.main()
