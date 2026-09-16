from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from agent.transaction_model import TransactionAgent
from agent.transaction_runtime import run_episode,ContextBudgetExceeded,SYSTEM,TOOLS
from agent.transaction_tasks import evaluate
from training.transaction_data import build_cases,profiles,teacher_trajectory,smoke_cases,SPLITS,canonical
from training.transaction_sft import encode_trajectory
from scripts.prepare_transaction_v1 import bundle,DATA,ROOT
from scripts.transaction_gpu_smoke import validate_records,run_meta,write_once


class TokenizerStub:
    eos_token_id=999999
    def apply_chat_template(self,messages,**kwargs):
        return canonical(messages)
    def encode(self,text,**kwargs):
        return [ord(c) for c in text]


class TransactionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=build_cases()

    def test_frozen_counts_and_structural_audit(self):
        m=bundle()
        self.assertEqual({s:m["counts"][s]["cases"] for s in SPLITS},
                         dict(train=2058,dev=256,confirmation_id=128,confirmation_ood=400))
        self.assertEqual(m["counts"]["train"]["assistant_examples"],31542)
        self.assertEqual(sum(m["counts"][s]["groups"] for s in SPLITS),98)
        self.assertEqual(m["counts"]["public_query_reuse"]["unique"],24)

    def test_group_partition_isolation_and_ood_mechanisms(self):
        groups={s:{c["group_id"] for c in self.cases if c["split"]==s} for s in SPLITS}
        for i,a in enumerate(SPLITS):
            for b in SPLITS[i+1:]:
                self.assertFalse(groups[a]&groups[b])
        for p in profiles():
            compound=p.get("timeout_after_commit",False) and p.get("concurrent",False)
            revoke=p.get("permission_case")=="revoke_before_commit"
            self.assertEqual(p["split"]=="confirmation_ood",compound or revoke)
        self.assertFalse((DATA/"confirmation_id_trajectories.jsonl").exists())
        self.assertFalse((DATA/"confirmation_ood_trajectories.jsonl").exists())

    def test_generation_deterministic_and_case_keys_unique(self):
        self.assertEqual(self.cases,build_cases())
        self.assertEqual(len({c["id"] for c in self.cases}),2842)
        self.assertEqual(len({canonical([c["environment"],c["desired"]]) for c in self.cases}),2842)

    def test_one_serialized_episode_per_group_matches_oracle(self):
        groups={c["group_id"]:c for c in self.cases}
        for c in groups.values():
            trajectory,source,_=teacher_trajectory(c)
            answers=iter(m["content"] for m in trajectory["messages"] if m["role"]=="assistant")
            def generate(messages,tools):
                self.assertEqual(tools,TOOLS)
                self.assertEqual(messages[:2],[dict(role="system",content=SYSTEM),dict(role="user",content=c["query"])])
                self.assertNotIn("concurrent_config",canonical(messages))
                self.assertNotIn("check_passes",canonical(messages))
                return next(answers)
            result=run_episode(generate,c["query"],c["environment"])
            self.assertEqual(result["events"],source["events"])
            self.assertTrue(evaluate(c["environment"],c["desired"],result["events"],result["final_answer"])["task_success"])

    def test_smoke_has_no_confirmation_and_is_not_formal_eval(self):
        rows=smoke_cases(self.cases)
        self.assertEqual(len(rows),14)
        self.assertEqual(Counter(r["split"] for r in rows),{"train":2,"dev":12})
        self.assertEqual(len({r["group_id"] for r in rows if r["split"]=="dev"}),12)
        self.assertTrue(all(r["category"]=="permission" for r in rows if r["split"]=="train"))

    def test_multi_turn_supervision_masks_observations(self):
        row,_,_=teacher_trajectory(next(c for c in self.cases if c["split"]=="train" and c["category"]=="rollback"))
        examples=list(encode_trajectory(row,TokenizerStub(),100000))
        targets=[m["content"] for m in row["messages"] if m["role"]=="assistant"]
        self.assertEqual(len(examples),len(targets))
        self.assertGreater(len(examples),5)
        for e,target in zip(examples,targets):
            self.assertEqual([v for v in e["labels"] if v!=-100],[ord(c) for c in target]+[999999])
            self.assertEqual(len(e["labels"]),len(e["input_ids"]))
        with self.assertRaises(ValueError):
            list(encode_trajectory(row,TokenizerStub(),16))

    def test_confirmation_encoder_and_schema_drift_rejected(self):
        row,_,_=teacher_trajectory(next(c for c in self.cases if c["split"]=="confirmation_id"))
        with self.assertRaises(ValueError):
            list(encode_trajectory(row,TokenizerStub(),100000))
        row["split"]="train"
        row["tools"]=[]
        with self.assertRaises(ValueError):
            list(encode_trajectory(row,TokenizerStub(),100000))

    def test_context_budget_is_explicit_not_silent_truncation(self):
        def generate(*_):
            raise ContextBudgetExceeded()
        c=self.cases[0]
        result=run_episode(generate,c["query"],c["environment"])
        self.assertEqual(result["terminated_reason"],"context_budget")
        self.assertEqual(result["events"],[])
        self.assertFalse(evaluate(c["environment"],c["desired"],result["events"],result["final_answer"])["task_success"])

    def test_gpu_backend_uses_new_tools_and_refuses_overlength_before_generation(self):
        agent=TransactionAgent.__new__(TransactionAgent)
        ids=SimpleNamespace(shape=(1,20))
        calls=[]
        agent.tokenizer=SimpleNamespace(apply_chat_template=lambda messages,**kw:(calls.append(kw) or {"input_ids":ids}))
        agent.max_length=25
        agent.max_new_tokens=10
        with self.assertRaises(ContextBudgetExceeded):
            agent.generate([dict(role="user",content="test")],TOOLS)
        self.assertEqual(calls[0]["tools"],TOOLS)

    def test_smoke_resume_replays_rejects_drift_and_duplicates(self):
        c=smoke_cases(self.cases)[0]
        trajectory,_,_=teacher_trajectory(c)
        answers=iter(m["content"] for m in trajectory["messages"] if m["role"]=="assistant")
        result=run_episode(lambda *_:next(answers),c["query"],c["environment"])
        meta=run_meta({"test":True},"base")
        record=dict(case=c,result=result,metadata=meta,
                    metrics=evaluate(c["environment"],c["desired"],result["events"],result["final_answer"]))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"base.jsonl"
            p.write_text(canonical(record)+"\n")
            self.assertEqual(len(validate_records(p,meta,[c])),1)
            with self.assertRaises(ValueError):
                validate_records(p,dict(meta,role="other"),[c])
            for invalid in (False,1):
                bad=deepcopy(record)
                bad["metrics"]["task_success"]=invalid
                p.write_text(canonical(bad)+"\n")
                with self.assertRaises(ValueError):
                    validate_records(p,meta,[c])
            p.write_text((canonical(record)+"\n")*2)
            with self.assertRaises(ValueError):
                validate_records(p,meta,[c])

    def test_preflight_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"meta.json"
            write_once(p,{"same":True})
            write_once(p,{"same":True})
            with self.assertRaises(ValueError):
                write_once(p,{"same":False})

    def test_shell_stops_before_smoke_training_when_base_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"scripts").mkdir()
            script=root/"scripts/superpod_transaction_smoke.sh"
            script.write_text((ROOT/"scripts/superpod_transaction_smoke.sh").read_text())
            fake=root/"python"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$TEST_CALLS\"\ncase \"$*\" in *'--stage base'*) exit 9;; esac\n")
            fake.chmod(0o755)
            log=root/"calls"
            result=subprocess.run(["bash",str(script)],capture_output=True,
                                  env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ["PATH"],TEST_CALLS=str(log)))
            self.assertEqual(result.returncode,9)
            self.assertIn("--stage base",log.read_text())
            self.assertNotIn("--stage train_smoke",log.read_text())
