import copy
import json
import unittest

from agent.transaction_env import TransactionEnvironment
from agent.transaction_tasks import oracle
from scripts.transaction_grpo_rollout_probe import schedule, validate_behavior, summarize
from training.transaction_grpo import advantages, strict_reward


class GRPOProbeTests(unittest.TestCase):
    def test_schedule_train_only_and_fixed(self):
        tasks = schedule()
        self.assertEqual(tasks, schedule())
        self.assertEqual(len(tasks), 32)
        self.assertEqual(len({t['case']['group_id'] for t in tasks}), 8)
        self.assertEqual(len({t['seed'] for t in tasks}), 32)
        self.assertTrue(all(t['case']['split'] == 'train' for t in tasks))
        self.assertEqual([sum(t['case']['category'] == c for t in tasks) for c in ('apply','rollback','permission')], [12,16,4])

    def test_zero_and_mixed_advantages(self):
        for rewards in ([1.,1.,1.,1.], [0.,0.,0.,0.]):
            self.assertEqual(advantages(rewards)['advantages'], [0.]*4)
        result = advantages([0.,0.,1.,1.])
        self.assertFalse(result['zero_variance'])
        self.assertAlmostEqual(sum(result['advantages']), 0.)
        self.assertLess(result['advantages'][0], 0)
        self.assertGreater(result['advantages'][2], 0)
        for invalid in ([float('nan'),1], [True,False], [1], [2,0]):
            with self.assertRaises(ValueError):
                advantages(invalid)

    def test_reward_requires_evidence(self):
        case = schedule()[0]['case']
        result = oracle(case['environment'], case['desired'])
        self.assertEqual(strict_reward(case, result)[0], 1.)
        fake = dict(events=[], final_answer=result['final_answer'], metrics={'task_success':True})
        self.assertEqual(strict_reward(case, fake)[0], 0.)
        records = [dict(case=case,result=result,reward=1.) for _ in range(4)]
        self.assertEqual(summarize(records)['zero_variance_groups'], 1)
        with self.assertRaises(ValueError):
            summarize(records[:3])
        records[0]['reward'] = 0.
        with self.assertRaises(ValueError):
            summarize(records)

    def test_behavior_validation(self):
        record = dict(result=dict(turns=[dict(model_output='x')]), behavior=[dict(
            prompt_ids=[1], generated_ids=[2], old_logprobs=[-.1], token_entropies=[.2], text='x')])
        validate_behavior(record)
        for key,value in [('old_logprobs',[float('nan')]), ('generated_ids',[]), ('text','y'), ('token_entropies',[-1.])]:
            bad = copy.deepcopy(record)
            bad['behavior'][0][key] = value
            with self.assertRaises(ValueError):
                validate_behavior(bad)

    def test_blocked_write_is_not_rewarded(self):
        case = next(t['case'] for t in schedule() if t['case']['category']=='permission')
        env = TransactionEnvironment(case['environment'])
        initial = env.execute(dict(name='inspect_workspace', arguments={}))['result']
        env.execute(dict(name='stage_config', arguments=dict(config=case['desired'], base_revision=initial['revision'])))
        final = env.execute(dict(name='inspect_workspace', arguments={}))['result']
        answer = json.dumps(dict(outcome='blocked', revision=final['revision'], config=final['config'], tool_errors=1, check_failures=0))
        reward, metrics = strict_reward(case, dict(events=env.audit()['events'], final_answer=answer))
        self.assertTrue(metrics['execution_success'])
        self.assertTrue(metrics['answer_correct'])
        self.assertEqual(reward, 0.)


if __name__ == '__main__':
    unittest.main()
