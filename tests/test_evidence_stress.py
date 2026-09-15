from collections import Counter
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.evidence_stress_v1 import verify_bundle, replay_prefix, run_rows, resume_ids, ROOT
from training.evidence_pilot import score, canonical
from agent.guarded_runtime import guarded_execute
from tools.executor import execute_tool_json


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.manifest = verify_bundle()

    def test_balanced_paired_structure(self):
        self.assertEqual(Counter(r["arm"] for r in self.rows),
                         {a:24 for a in ("original","surface","extended_history","current_policy_flip")})
        self.assertEqual(len({r["context_id"] for r in self.rows}),24)
        self.assertEqual(len({r["id"] for r in self.rows}),96)
        for r in self.rows:
            self.assertEqual(r["split"],"post_training_diagnostic_not_heldout")

    def test_metamorphic_relations(self):
        originals={r["context_id"]:r for r in self.rows if r["arm"]=="original"}
        for r in self.rows:
            b=originals[r["context_id"]]
            if r["arm"]=="surface":
                self.assertEqual(r["entries"],b["entries"])
                self.assertEqual(r["target"],b["target"])
                self.assertNotEqual(r["query"],b["query"])
            elif r["arm"]=="current_policy_flip":
                self.assertEqual(r["entries"],b["entries"])
                self.assertNotEqual(r["target"]["decision"],b["target"]["decision"])
                self.assertEqual({k:v for k,v in r["target"].items() if k!="decision"},
                                 {k:v for k,v in b["target"].items() if k!="decision"})
            elif r["arm"]=="extended_history":
                self.assertFalse(score(canonical(b["target"]),r["target"])["exact_report"])
                self.assertGreater(len(r["entries"]),len(b["entries"]))

    def test_replay_all_histories_with_historical_policy(self):
        for r in self.rows:
            c=deepcopy(r)
            c["read_only"]=c["historical_read_only"]
            replay_prefix(c)

    def test_independent_target_counts_and_latest_evidence(self):
        for r in self.rows:
            results=r["entries"]
            errors=sum(e["result"]["ok"] is False for e in results)
            verifications=[e["result"]["result"] for e in results
                           if e["tool_call"]["name"]=="verify_state" and e["result"]["ok"]]
            t=r["target"]
            self.assertEqual(t["total_failures"],errors+sum(v["success"] is False for v in verifications))
            self.assertEqual(t["last_verification_matches"],verifications[-1]["matches"])
            observed=[e["result"]["result"] for e in results
                      if e["tool_call"]["name"]=="inspect_ui_state" and e["result"]["ok"]]
            self.assertEqual(t["state"],{k:observed[-1][k] for k in t["state"]})
            self.assertTrue(score(canonical(t),t)["exact_report"])

    def test_model_only_receives_public_query(self):
        c=self.rows[0]
        seen=[]
        def generate(messages):
            seen.append(messages)
            return canonical(c["target"])
        handle=io.StringIO()
        run_rows(generate,[c],{"test":True},handle,"FAKE TEST GPU")
        self.assertEqual(seen[0][1],{"role":"user","content":c["query"]})
        row=json.loads(handle.getvalue())
        self.assertTrue(row["metrics"]["exact_report"])

    def test_resume_drift_and_duplicates(self):
        c=self.rows[0]
        row=dict(case=c,answer=canonical(c["target"]),metrics=score(canonical(c["target"]),c["target"]),
                 metadata={"test":True})
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"output.jsonl"
            p.write_text(canonical(row)+"\n")
            self.assertEqual(resume_ids(p,{"test":True},[c]),{c["id"]})
            with self.assertRaises(ValueError):
                resume_ids(p,{"test":False},[c])
            p.write_text((canonical(row)+"\n")*2)
            with self.assertRaises(ValueError):
                resume_ids(p,{"test":True},[c])

    def test_shell_stops_before_second_model_on_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"scripts").mkdir()
            script=root/"scripts/superpod_evidence_stress.sh"
            script.write_text((ROOT/"scripts/superpod_evidence_stress.sh").read_text())
            fake=root/"python"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$TEST_CALLS\"\ncase \"$*\" in *'--role base'*) exit 9;; esac\n")
            fake.chmod(0o755)
            log=root/"calls"
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ["PATH"],TEST_CALLS=str(log))
            result=subprocess.run(["bash",str(script)],env=env,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("--role base",log.read_text())
            self.assertNotIn("--role new_sft",log.read_text())
