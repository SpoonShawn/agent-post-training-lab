from collections import Counter
import unittest
from scripts.transaction_preference_probe import slots, pair_indices


class ProbeTests(unittest.TestCase):
    def test_train_only_balanced_deterministic_slots(self):
        selected = slots()
        self.assertEqual(selected, slots())
        self.assertEqual(len(selected), 392)
        self.assertEqual({s["case"]["split"] for s in selected}, {"train"})
        self.assertEqual(len({s["case"]["group_id"] for s in selected}), 49)
        self.assertEqual(set(Counter(s["case"]["id"] for s in selected).values()), {4})
        self.assertEqual(len({s["seed"] for s in selected}), 392)

    def test_abstains_on_ties_and_does_not_pair_across_tasks(self):
        def row(cid, success):
            return dict(case=dict(id=cid), metrics=dict(task_success=success))
        result = pair_indices([row("a", True), row("b", False), row("c", False), row("c", True)])
        self.assertEqual(result["abstained_task_ids"], ["a", "b"])
        self.assertEqual(result["pairs"], [dict(case_id="c", chosen_index=3, rejected_index=2)])


if __name__ == "__main__":
    unittest.main()
