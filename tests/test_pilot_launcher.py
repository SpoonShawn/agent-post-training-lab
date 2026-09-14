import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.run_baseline import parse_args

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def launch(self, fail_baseline=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            script = root / "scripts/superpod_train_pilot.sh"
            script.write_text((ROOT / "scripts/superpod_train_pilot.sh").read_text())
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "python"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['ARGUMENT_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if os.environ.get('FAIL_BASELINE') == '1' and 'scripts/run_baseline.py' in sys.argv: sys.exit(2)\n")
            fake.chmod(0o755)
            log = root / "argv.jsonl"
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       ARGUMENT_LOG=str(log), FAIL_BASELINE="1" if fail_baseline else "0")
            process = subprocess.run(["bash", str(script)], cwd=tmp, env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return process, calls

    def test_actual_shell_arguments_parse_without_stray_tokens(self):
        process, calls = self.launch()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(calls), 8)
        for args in calls:
            self.assertNotIn("+", args)
            if "scripts/run_baseline.py" in args:
                parse_args(args[args.index("scripts/run_baseline.py") + 1:])
            if "scripts/train_sft.py" in args:
                rest = args[args.index("scripts/train_sft.py") + 1:]
                self.assertIn("--output-dir", rest)
                self.assertTrue(set(rest[::2]) <= {"--max-steps", "--output-dir"})
        self.assertIn("[2/7]", process.stdout)
        self.assertIn("[7/7]", process.stdout)

    def test_failed_baseline_stops_before_training(self):
        process, calls = self.launch(fail_baseline=True)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(len(calls), 3)
        self.assertFalse(any("scripts/train_sft.py" in c for c in calls))
