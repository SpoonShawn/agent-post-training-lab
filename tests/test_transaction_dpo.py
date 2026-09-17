import math
import unittest

from scripts.prepare_transaction_dpo import build
from training.transaction_dpo import dpo_terms, encode_pair, trajectory_row


class Tokenizer:
    eos_token_id = 99999
    def apply_chat_template(self, messages, **kwargs):
        return "|".join(m["role"]+":"+m["content"] for m in messages) + "assistant:"
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class DPOTests(unittest.TestCase):
    def test_real_pairs_replay_and_group_split(self):
        records, summary = build()
        self.assertEqual(summary["pairs"], 98)
        self.assertEqual(summary["base_success"], 0)
        self.assertEqual(summary["splits"], {"preference_train":80, "preference_dev":18})
        train = {r["group_id"] for r in records if r["split"] == "preference_train"}
        dev = {r["group_id"] for r in records if r["split"] == "preference_dev"}
        self.assertFalse(train & dev)
        self.assertEqual((len(train), len(dev)), (40, 9))
        self.assertEqual({r["case"]["split"] for r in records}, {"train"})

    def test_dpo_initial_loss_and_finite_difference(self):
        # Low rejected likelihood does NOT saturate initial DPO: reference ratio cancels.
        loss, derivative, margin = dpo_terms(-.01, -5000, -.01, -5000)
        self.assertAlmostEqual(loss, math.log(2))
        self.assertAlmostEqual(derivative, -.05)
        self.assertEqual(margin, 0)
        epsilon = 1e-5
        plus = dpo_terms(-1+epsilon, -5, -2, -6)[0]
        minus = dpo_terms(-1-epsilon, -5, -2, -6)[0]
        self.assertAlmostEqual((plus-minus)/(2*epsilon), dpo_terms(-1, -5, -2, -6)[1], places=7)
        for value in (-1e6, 1e6):
            self.assertTrue(all(math.isfinite(v) for v in dpo_terms(value, 0, 0, 0)))

    def test_multiturn_masks_and_own_tool_observations(self):
        result = dict(query="goal", terminated_reason="final_answer", turns=[
            dict(model_output="action", result={"ok":True,"result":{"value":"OBSERVATION"}}),
            dict(model_output="report")])
        row = trajectory_row("id", result)
        pair = dict(case_id="id", chosen=result, rejected=result)
        encoded = encode_pair(pair, Tokenizer())["chosen"]
        self.assertEqual(len(encoded), 2)
        for example, target in zip(encoded, ("action", "report")):
            labels = example["labels"]
            self.assertEqual([x for x in labels if x != -100], [ord(c) for c in target]+[99999])
        prefix = encoded[1]["input_ids"][:-len("report")-1]
        self.assertIn("OBSERVATION", "".join(chr(c) for c in prefix))
        self.assertEqual(row["messages"][3]["role"], "tool")
        result["terminated_reason"] = "max_calls"
        with self.assertRaises(ValueError):
            trajectory_row("id", result)


if __name__ == "__main__":
    unittest.main()
