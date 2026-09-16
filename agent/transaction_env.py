"""Isolated BugOps configuration transaction sandbox, separate from frozen pilots.

No network/filesystem effects. Scenario controls are evaluator-only; public tools
return observations, never the fault schedule or hidden outcome configuration.
"""
from copy import deepcopy
import json


class ToolFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise ToolFailure(code, message)


def config_ok(config):
    return (isinstance(config, dict) and set(config) == {"quality", "cache_enabled"}
            and config["quality"] in ("low", "standard", "high")
            and type(config["cache_enabled"]) is bool)


class TransactionEnvironment:
    def __init__(self, scenario=None):
        scenario = deepcopy(scenario or {})
        allowed = {"initial_config", "revision", "writable", "transient_commits",
                   "timeout_after_commit", "concurrent_config", "revoke_before_commit",
                   "check_delay", "check_passes"}
        if set(scenario) - allowed:
            raise ValueError("Unknown scenario fields")
        self.config = scenario.get("initial_config", {"quality":"high", "cache_enabled":True})
        if not config_ok(self.config):
            raise ValueError("Invalid initial config")
        self.revision = scenario.get("revision", 1)
        self.writable = scenario.get("writable", True)
        self._transient = scenario.get("transient_commits", 0)
        self._timeout = scenario.get("timeout_after_commit", False)
        self._concurrent = scenario.get("concurrent_config")
        self._revoke = scenario.get("revoke_before_commit", False)
        self._delay = scenario.get("check_delay", 1)
        self._check_passes = scenario.get("check_passes", True)
        for value in (self.writable, self._timeout, self._revoke, self._check_passes):
            if type(value) is not bool:
                raise ValueError("Scenario flags must be boolean")
        for value, minimum in ((self.revision,1),(self._transient,0),(self._delay,0)):
            if type(value) is not int or value < minimum:
                raise ValueError("Invalid scenario integer")
        if self._concurrent is not None and not config_ok(self._concurrent):
            raise ValueError("Invalid concurrent config")
        self._candidates, self._operations, self._jobs = {}, {}, {}
        self._mutations, self._events = [], []

    def inspect_workspace(self):
        return dict(config=deepcopy(self.config), revision=self.revision, writable=self.writable)

    def stage_config(self, config, base_revision):
        require(config_ok(config), "invalid_arguments", "config需含quality枚举和布尔cache_enabled")
        self._revision_arg(base_revision)
        self._permission()
        require(base_revision == self.revision, "stale_revision", "请重新读取当前revision，再暂存候选")
        cid = f"candidate_{len(self._candidates)+1}"
        self._candidates[cid] = dict(config=deepcopy(config), base_revision=base_revision, validated=False)
        return dict(candidate_id=cid, base_revision=base_revision, config=deepcopy(config))

    def validate_candidate(self, candidate_id):
        candidate = self._candidate(candidate_id)
        require(candidate["base_revision"] == self.revision, "stale_revision", "候选基于过期revision，需重新暂存")
        candidate["validated"] = True
        return dict(candidate_id=candidate_id, valid=True, base_revision=candidate["base_revision"])

    def commit_candidate(self, candidate_id, operation_id):
        signature = dict(kind="commit", candidate_id=candidate_id)
        cached = self._cached(operation_id, signature)
        if cached is not None:
            return cached
        candidate = self._candidate(candidate_id)
        require(candidate["validated"], "not_validated", "提交前必须validate_candidate成功")
        self._permission()
        if self._revoke:
            self._revoke = False
            self.writable = False
            self._permission()
        if self._concurrent is not None:
            self.config = deepcopy(self._concurrent)
            self.revision += 1
            self._concurrent = None
        require(candidate["base_revision"] == self.revision, "stale_revision", "并发修改使revision过期；重新读取并暂存，不可覆盖旧revision")
        if self._transient:
            self._transient -= 1
            raise ToolFailure("transient_unavailable", "此次提交尚未执行，可确认状态后重试")
        before = self.inspect_workspace()
        self.config = deepcopy(candidate["config"])
        self.revision += 1
        receipt = dict(operation_id=operation_id, status="committed", revision=self.revision,
                       config=deepcopy(self.config), previous_config=before["config"],
                       previous_revision=before["revision"])
        self._operations[operation_id] = dict(signature=signature, receipt=receipt)
        self._mutations.append(dict(operation_id=operation_id, kind="commit", revision=self.revision))
        if self._timeout:
            self._timeout = False
            raise ToolFailure("outcome_unknown", "提交响应超时，结果未知；用get_operation查询该operation_id，不要换ID重复提交")
        return deepcopy(receipt)

    def get_operation(self, operation_id):
        self._identifier(operation_id)
        record = self._operations.get(operation_id)
        return (deepcopy(record["receipt"]) if record else dict(operation_id=operation_id, status="not_found"))

    def start_check(self, revision):
        self._revision_arg(revision)
        require(revision == self.revision, "stale_revision", "只能对当前revision启动检查")
        jid = f"job_{len(self._jobs)+1}"
        self._jobs[jid] = dict(revision=revision, config=deepcopy(self.config), remaining=self._delay)
        return dict(job_id=jid, status="pending", revision=revision)

    def poll_check(self, job_id):
        self._identifier(job_id)
        require(job_id in self._jobs, "not_found", "不存在该检查任务")
        job = self._jobs[job_id]
        if job["remaining"]:
            job["remaining"] -= 1
            return dict(job_id=job_id, status="pending", revision=job["revision"])
        return dict(job_id=job_id, status="passed" if self._check_passes else "failed",
                    revision=job["revision"], config=deepcopy(job["config"]))

    def rollback(self, committed_operation_id, expected_revision, operation_id):
        self._revision_arg(expected_revision)
        self._identifier(committed_operation_id)
        signature = dict(kind="rollback", committed_operation_id=committed_operation_id,
                         expected_revision=expected_revision)
        cached = self._cached(operation_id, signature)
        if cached is not None:
            return cached
        self._permission()
        record = self._operations.get(committed_operation_id)
        require(record is not None and record["signature"]["kind"] == "commit", "not_found", "需指定成功提交的operation_id")
        receipt = record["receipt"]
        require(expected_revision == self.revision == receipt["revision"], "stale_revision", "不能回滚覆盖后续修改；请读取当前状态")
        self.config = deepcopy(receipt["previous_config"])
        self.revision += 1
        result = dict(operation_id=operation_id, status="rolled_back", revision=self.revision,
                      config=deepcopy(self.config), reverted_operation_id=committed_operation_id)
        self._operations[operation_id] = dict(signature=signature, receipt=result)
        self._mutations.append(dict(operation_id=operation_id, kind="rollback", revision=self.revision))
        return deepcopy(result)

    @staticmethod
    def _identifier(value):
        require(isinstance(value,str) and 0 < len(value) <= 80, "invalid_arguments", "标识符必须为1至80字符的字符串")

    @staticmethod
    def _revision_arg(value):
        require(type(value) is int and value >= 1, "invalid_arguments", "revision必须为正整数，不能是boolean")

    def _permission(self):
        require(self.writable, "permission_denied", "当前禁止写入；停止修改并报告阻塞，读取工具仍可用")

    def _candidate(self, cid):
        self._identifier(cid)
        require(cid in self._candidates, "not_found", "不存在该候选")
        return self._candidates[cid]

    def _cached(self, operation_id, signature):
        self._identifier(operation_id)
        previous = self._operations.get(operation_id)
        if previous:
            require(previous["signature"] == signature, "idempotency_conflict", "同一operation_id不能用于不同请求")
            return dict(deepcopy(previous["receipt"]), replayed=True)
        return None

    def execute(self, call):
        allowed = {"inspect_workspace", "stage_config", "validate_candidate", "commit_candidate",
                   "get_operation", "start_check", "poll_check", "rollback"}
        try:
            require(isinstance(call,dict) and set(call) == {"name","arguments"}, "invalid_call", "调用必须且只能含name和arguments")
            name, args = call["name"], call["arguments"]
            require(isinstance(name,str) and name in allowed, "unknown_tool", "未知工具")
            require(isinstance(args,dict), "invalid_arguments", "arguments必须是object")
            result = dict(ok=True, result=getattr(self,name)(**deepcopy(args)))
        except ToolFailure as exc:
            result = dict(ok=False, error_type=exc.code, error=str(exc))
        except TypeError:
            result = dict(ok=False, error_type="invalid_arguments", error="参数名称、数量或类型错误")
        # Unexpected implementation exceptions propagate; never label a crash as model failure.
        result = deepcopy(result)
        self._events.append(dict(index=len(self._events), tool_call=deepcopy(call), result=deepcopy(result)))
        return result

    def audit(self):
        """Evaluator-only; do NOT pass this object to the policy."""
        return deepcopy(dict(events=self._events, mutations=self._mutations, final_state=self.inspect_workspace()))

    def snapshot(self):
        """Evaluator-only snapshot for branched rollouts; not a model tool."""
        return deepcopy(self)


def replay(scenario, events):
    env = TransactionEnvironment(scenario)
    for index, entry in enumerate(events):
        if (not isinstance(entry,dict) or set(entry) != {"index","tool_call","result"}
                or type(entry["index"]) is not int or entry["index"] != index):
            raise ValueError(f"Trace mismatch at event {index}")
        # Python equality considers True == 1. Preserve JSON types during evidence validation.
        actual = env.execute(entry["tool_call"])
        if json.dumps(actual,sort_keys=True) != json.dumps(entry["result"],sort_keys=True):
            raise ValueError(f"Trace mismatch at event {index}")
    return env.audit()
