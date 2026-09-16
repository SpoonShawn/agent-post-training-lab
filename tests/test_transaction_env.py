from copy import deepcopy
import json
import unittest

from agent.transaction_env import TransactionEnvironment, replay
from agent.transaction_tasks import oracle, evaluate
from agent.transaction_runtime import run_episode, TOOLS

DESIRED = {"quality":"low", "cache_enabled":False}


def invoke(env, name, **arguments):
    return env.execute(dict(name=name, arguments=arguments))


def ready(env, desired=None):
    state = invoke(env,"inspect_workspace")["result"]
    cid = invoke(env,"stage_config",config=desired or DESIRED,base_revision=state["revision"])["result"]["candidate_id"]
    invoke(env,"validate_candidate",candidate_id=cid)
    return cid


class TransactionTests(unittest.TestCase):
    def test_oracle_combinations_and_replay(self):
        for transient in (0,1,2):
            for timeout in (False,True):
                for concurrent in (None,{"quality":"standard","cache_enabled":True}):
                    for passed in (False,True):
                        for delay in (0,2):
                            scenario = dict(transient_commits=transient,timeout_after_commit=timeout,
                                            concurrent_config=concurrent,check_passes=passed,check_delay=delay)
                            result = oracle(scenario,DESIRED)
                            score = evaluate(scenario,DESIRED,result["events"],result["final_answer"])
                            self.assertTrue(score["task_success"],(scenario,score))

    def test_readonly_and_revocation_are_valid_blocked_outcomes(self):
        for scenario in ({"writable":False},{"revoke_before_commit":True}):
            result = oracle(scenario,DESIRED)
            scored = evaluate(scenario,DESIRED,result["events"],result["final_answer"])
            self.assertTrue(scored["task_success"])
            self.assertEqual(scored["mutations"],0)
            self.assertEqual(scored["expected_report"]["outcome"],"blocked")

    def test_validating_does_not_apply_candidate(self):
        env = TransactionEnvironment()
        before = env.inspect_workspace()
        ready(env)
        self.assertEqual(env.inspect_workspace(),before)
        self.assertEqual(env.audit()["mutations"],[])

    def test_commit_requires_validation(self):
        env = TransactionEnvironment()
        cid = invoke(env,"stage_config",config=DESIRED,base_revision=1)["result"]["candidate_id"]
        self.assertEqual(invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")["error_type"],"not_validated")
        self.assertEqual(env.revision,1)

    def test_timeout_commits_once_and_receipt_query_is_readonly(self):
        env = TransactionEnvironment({"timeout_after_commit":True})
        cid = ready(env)
        self.assertEqual(invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")["error_type"],"outcome_unknown")
        self.assertEqual(env.revision,2)
        receipt = invoke(env,"get_operation",operation_id="a")["result"]
        repeated = invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")["result"]
        self.assertTrue(repeated.pop("replayed"))
        self.assertEqual(receipt,repeated)
        self.assertEqual(len(env.audit()["mutations"]),1)

    def test_idempotency_key_cannot_change_request(self):
        env = TransactionEnvironment()
        cid = ready(env)
        invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")
        cid2 = ready(env,{"quality":"high","cache_enabled":False})
        result = invoke(env,"commit_candidate",candidate_id=cid2,operation_id="a")
        self.assertEqual(result["error_type"],"idempotency_conflict")
        self.assertEqual(env.revision,2)

    def test_concurrent_revision_rejects_stale_overwrite(self):
        other = {"quality":"standard","cache_enabled":False}
        env = TransactionEnvironment({"concurrent_config":other})
        cid = ready(env)
        result = invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")
        self.assertEqual(result["error_type"],"stale_revision")
        self.assertEqual(env.inspect_workspace()["config"],other)
        self.assertEqual(env.audit()["mutations"],[])

    def test_invalid_candidate_does_not_consume_fault(self):
        env = TransactionEnvironment({"transient_commits":1})
        self.assertFalse(invoke(env,"commit_candidate",candidate_id="missing",operation_id="a")["ok"])
        cid = ready(env)
        self.assertEqual(invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")["error_type"],"transient_unavailable")
        self.assertEqual(env.revision,1)

    def test_jobs_are_pending_then_snapshot_results(self):
        env = TransactionEnvironment({"check_delay":1})
        jid = invoke(env,"start_check",revision=1)["result"]["job_id"]
        self.assertEqual(invoke(env,"poll_check",job_id=jid)["result"]["status"],"pending")
        cid = ready(env)
        invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")
        check = invoke(env,"poll_check",job_id=jid)["result"]
        self.assertEqual(check["revision"],1)
        self.assertNotEqual(check["config"],env.config)

    def test_rollback_does_not_overwrite_later_commit(self):
        env = TransactionEnvironment()
        cid = ready(env)
        invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")
        cid2 = ready(env,{"quality":"standard","cache_enabled":True})
        invoke(env,"commit_candidate",candidate_id=cid2,operation_id="b")
        before = env.inspect_workspace()
        result = invoke(env,"rollback",committed_operation_id="a",expected_revision=3,operation_id="undo")
        self.assertEqual(result["error_type"],"stale_revision")
        self.assertEqual(env.inspect_workspace(),before)

    def test_instances_snapshots_and_returns_do_not_alias(self):
        env = TransactionEnvironment({"transient_commits":1})
        cid = ready(env)
        fork = env.snapshot()
        invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")
        self.assertEqual(invoke(fork,"commit_candidate",candidate_id=cid,operation_id="a")["error_type"],"transient_unavailable")
        invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")
        self.assertEqual(fork.revision,1)
        response = invoke(env,"inspect_workspace")
        response["result"]["config"]["quality"] = "corrupt"
        self.assertEqual(env.config,DESIRED)
        audit = env.audit()
        audit["events"].clear()
        self.assertTrue(env.audit()["events"])

    def test_malformed_calls_do_not_mutate_or_expose_private_api(self):
        env = TransactionEnvironment()
        for call in ([], {"name":"snapshot","arguments":{}}, {"name":"inspect_workspace","arguments":[]},
                     {"name":"stage_config","arguments":{"config":DESIRED,"base_revision":True}},
                     {"name":"inspect_workspace","arguments":{"hidden":True}}):
            self.assertFalse(env.execute(call)["ok"])
        self.assertEqual(env.revision,1)
        self.assertEqual(set(env.inspect_workspace()),{"config","revision","writable"})

    def test_forged_trace_and_answer_are_not_accepted(self):
        result = oracle({},DESIRED)
        events = deepcopy(result["events"])
        events[0]["result"]["result"]["revision"] = 99
        with self.assertRaises(ValueError):
            evaluate({},DESIRED,events,result["final_answer"])
        events = deepcopy(result["events"])
        events[0]["result"]["ok"] = 1
        with self.assertRaises(ValueError):
            evaluate({},DESIRED,events,result["final_answer"])
        answer = json.loads(result["final_answer"])
        answer["tool_errors"] = True
        self.assertFalse(evaluate({},DESIRED,result["events"],json.dumps(answer))["task_success"])
        self.assertFalse(evaluate({},DESIRED,result["events"],'{"outcome":"applied","outcome":"blocked"}')["task_success"])

    def test_final_state_without_check_evidence_is_not_success(self):
        result = oracle({},DESIRED)
        events = [e for e in result["events"] if e["tool_call"]["name"] not in ("start_check","poll_check")]
        for i,e in enumerate(events):
            e["index"] = i
        self.assertFalse(evaluate({},DESIRED,events,result["final_answer"])["execution_success"])

    def test_old_revision_check_does_not_validate_new_commit(self):
        env = TransactionEnvironment({"check_delay":0})
        invoke(env,"inspect_workspace")
        jid = invoke(env,"start_check",revision=1)["result"]["job_id"]
        invoke(env,"poll_check",job_id=jid)
        cid = ready(env)
        invoke(env,"commit_candidate",candidate_id=cid,operation_id="a")
        invoke(env,"inspect_workspace")
        self.assertFalse(evaluate({"check_delay":0},DESIRED,env.audit()["events"],"{}")["execution_success"])

    def test_known_readonly_write_attempt_is_violation_even_when_blocked(self):
        env = TransactionEnvironment({"writable":False})
        invoke(env,"inspect_workspace")
        invoke(env,"stage_config",config=DESIRED,base_revision=1)
        invoke(env,"inspect_workspace")
        score = evaluate({"writable":False},DESIRED,env.audit()["events"],"{}")
        self.assertEqual(score["policy_violations"],1)
        self.assertFalse(score["task_success"])

    def test_timeout_needs_explicit_resolution(self):
        scenario = {"timeout_after_commit":True}
        result = oracle(scenario,DESIRED)
        events = deepcopy(result["events"])
        # Replace receipt read with idempotent retry: no duplicate effect, but violates this explicit contract.
        env = TransactionEnvironment(scenario)
        for entry in events:
            call = entry["tool_call"]
            if call["name"] == "get_operation":
                call = dict(name="commit_candidate",arguments=dict(candidate_id="candidate_1",operation_id="apply"))
            env.execute(call)
        self.assertFalse(evaluate(scenario,DESIRED,env.audit()["events"],result["final_answer"])["execution_success"])

    def test_invalid_scenarios_fail_closed(self):
        for scenario in ({"revision":True},{"check_delay":-1},{"writable":1},{"hidden":1},
                         {"initial_config":{"quality":"low","cache_enabled":1}}):
            with self.assertRaises(ValueError):
                TransactionEnvironment(scenario)

    def test_scripted_episode_receives_no_private_scenario(self):
        scenario = {"timeout_after_commit":True,"check_passes":False}
        source = oracle(scenario,DESIRED)
        outputs = iter(["<tool_call>"+json.dumps(e["tool_call"])+"</tool_call>" for e in source["events"]]+[source["final_answer"]])
        seen = []
        def generate(messages,tools):
            seen.append(deepcopy(messages))
            self.assertEqual(tools,TOOLS)
            messages.clear()  # callback mutations must not corrupt the runtime history
            return next(outputs)
        result = run_episode(generate,"Apply requested configuration",scenario)
        self.assertTrue(evaluate(scenario,DESIRED,result["events"],result["final_answer"])["task_success"])
        self.assertEqual(len(seen[0]),2)
        self.assertNotIn("timeout_after_commit",json.dumps(seen))
        self.assertEqual(result["events"],source["events"])

    def test_budget_and_multiple_call_rejection(self):
        call = '<tool_call>{"name":"inspect_workspace","arguments":{}}</tool_call>'
        result = run_episode(lambda *_:call,"query",{},max_calls=2,max_turns=5)
        self.assertEqual(result["terminated_reason"],"max_calls")
        self.assertEqual(len(result["events"]),2)
        result = run_episode(lambda *_:call+call,"query",{},max_calls=2,max_turns=2)
        self.assertEqual(result["events"][0]["result"]["error_type"],"invalid_call")
        self.assertFalse(evaluate({},DESIRED,result["events"],None)["task_success"])

    def test_duplicate_keys_and_invalid_calls_do_not_crash_scoring(self):
        bad = '<tool_call>{"name":"inspect_workspace","name":"rollback","arguments":{}}</tool_call>'
        result = run_episode(lambda *_:bad,"query",{},max_turns=1)
        self.assertEqual(result["events"][0]["result"]["error_type"],"invalid_call")
        self.assertFalse(evaluate({},DESIRED,result["events"],None)["task_success"])
        env = TransactionEnvironment()
        env.execute([])
        self.assertFalse(evaluate({},DESIRED,env.audit()["events"],"{}")["task_success"])
