"""Deterministic infrastructure probe, not an implementation of an autonomous agent.

All authoritative probe state lives on STATE_BRANCH; NRC product files are untouched.
Only the controller writes that state. Workers have read-only repository permissions.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SOURCE_BRANCH = "codex/github-agent-validation"
STATE_BRANCH = "codex-agent-probe-state"
STATE_PATH = "validation-state/ledger.json"
CONTROLLER = "agent-probe-controller.yml"
WORKER = "agent-probe-worker.yml"


class APIError(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        super().__init__(f"GitHub API {status}: {detail[:350]}")


def scenario_id(value):
    if not re.fullmatch(r"probe-[A-Za-z0-9-]{1,80}", value):
        raise ValueError("Invalid scenario identifier")
    return value


def new_ledger():
    return {"schema_version": 1, "scenarios": {}}


def prepare(ledger, sid, sha):
    scenario_id(sid)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected an immutable source commit")
    if sid in ledger["scenarios"]:
        return "already_prepared"
    ledger["scenarios"][sid] = {
        "phase": "ready", "goal_revision": 1, "source_sha": sha,
        "task_id": sid + "-multiply", "attempt_id": 1,
        "task": {"operation": "multiply", "operands": [6, 7]},
        "dispatch": {"state": "pending", "key": "dispatch:" + sid},
        "accepted_result": None, "duplicate_deliveries": 0,
        "rejected_results": [],
        "events": [{"id": "prepare:" + sid, "type": "task.prepared"}],
    }
    return "prepared"


def result_for(sid, item, worker_run_id):
    return {
        "scenario_id": sid, "task_id": item["task_id"],
        "attempt_id": item["attempt_id"], "goal_revision": item["goal_revision"],
        "source_sha": item["source_sha"], "worker_run_id": worker_run_id,
        "value": item["task"]["operands"][0] * item["task"]["operands"][1],
    }


def result_key(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def apply_result(ledger, result):
    sid = scenario_id(result.get("scenario_id", ""))
    item = ledger["scenarios"][sid]
    mismatches = [key for key in ("task_id", "attempt_id", "goal_revision", "source_sha")
                  if result.get(key) != item[key]]
    if result.get("value") != 42 or type(result.get("value")) is not int:
        mismatches.append("value")
    digest = result_key(result)
    if mismatches:
        if not any(x["digest"] == digest for x in item["rejected_results"]):
            item["rejected_results"].append({"digest": digest, "reasons": mismatches})
        return "rejected"
    existing = item["accepted_result"]
    if existing is not None:
        if existing != result:
            raise ValueError("Conflicting successful result for the same attempt")
        item["duplicate_deliveries"] += 1
        return "duplicate"
    item["accepted_result"] = copy.deepcopy(result)
    item["phase"] = "completed"
    item["events"].append({"id": "result:" + sid, "type": "task.completed"})
    return "accepted"


def validate_worker_run(run, jobs, item, sid):
    if run.get("path", "").split("@")[0] != ".github/workflows/" + WORKER:
        raise ValueError("Result did not originate from the worker workflow")
    if run.get("event") != "workflow_dispatch":
        raise ValueError("Worker event was not workflow_dispatch")
    if run.get("head_sha") != item["source_sha"]:
        raise ValueError("Worker source commit differs from the authorized task")
    if run.get("display_title") != "Probe worker / " + sid:
        raise ValueError("Worker run belongs to another scenario")
    if not any(j.get("name") == "deterministic-worker" and
               j.get("status") == "completed" and j.get("conclusion") == "success"
               for j in jobs):
        raise ValueError("Worker execution job has no successful completion evidence")


class GitHub:
    def __init__(self):
        self.repo = os.environ["GITHUB_REPOSITORY"]
        self.token = os.environ["GITHUB_TOKEN"]
        self.prefix = "/repos/" + self.repo

    def api(self, method, path, payload=None):
        encoded = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            "https://api.github.com" + self.prefix + path,
            data=encoded, method=method,
            headers={"Authorization": "Bearer " + self.token,
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json", "User-Agent": "NRC-agent-probe"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            raise APIError(exc.code, exc.read().decode(errors="replace")) from None

    def ensure_state_branch(self):
        try:
            self.api("GET", "/git/ref/heads/" + STATE_BRANCH)
        except APIError as exc:
            if exc.status != 404:
                raise
            head = self.api("GET", "/git/ref/heads/" + SOURCE_BRANCH)["object"]["sha"]
            try:
                self.api("POST", "/git/refs", {"ref": "refs/heads/" + STATE_BRANCH, "sha": head})
            except APIError as race:
                if race.status != 422:
                    raise
                self.api("GET", "/git/ref/heads/" + STATE_BRANCH)

    def read_state(self):
        try:
            record = self.api("GET", "/contents/" + STATE_PATH + "?ref=" + STATE_BRANCH)
            return json.loads(base64.b64decode(record["content"])), record["sha"]
        except APIError as exc:
            if exc.status == 404:
                return new_ledger(), None
            raise

    def mutate(self, change):
        self.ensure_state_branch()
        for attempt in range(8):
            ledger, sha = self.read_state()
            outcome = change(ledger)
            payload = {
                "message": "Record agent infrastructure probe state",
                "branch": STATE_BRANCH,
                "content": base64.b64encode((json.dumps(ledger, indent=2) + "\n").encode()).decode(),
            }
            if sha:
                payload["sha"] = sha
            try:
                self.api("PUT", "/contents/" + STATE_PATH, payload)
                return outcome
            except APIError as exc:
                if exc.status not in (409, 422) or attempt == 7:
                    raise
                time.sleep(0.5 + attempt * 0.4)
        raise RuntimeError("State update retries exhausted")

    def dispatch(self, workflow, inputs):
        return self.api("POST", "/actions/workflows/" + workflow + "/dispatches",
                        {"ref": SOURCE_BRANCH, "inputs": inputs})

    def worker_run_for(self, sid):
        query = urllib.parse.urlencode({"event": "workflow_dispatch", "branch": SOURCE_BRANCH,
                                        "per_page": 100})
        records = self.api("GET", "/actions/workflows/" + WORKER + "/runs?" + query)
        return next((r for r in records["workflow_runs"]
                     if r.get("display_title") == "Probe worker / " + sid), None)


def summary(value):
    text = json.dumps(value, indent=2)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf8") as file:
            file.write("### Infrastructure probe\n\n```json\n" + text + "\n```\n")


def controller():
    gh = GitHub()
    phase = os.environ.get("PROBE_PHASE") or "start"
    if phase == "ingest":
        raw = os.environ["PROBE_RESULT"]
        if len(raw) > 4000:
            raise ValueError("Oversized result")
        result = json.loads(raw)
        sid = scenario_id(result.get("scenario_id", ""))
        ledger, _ = gh.read_state()
        item = ledger["scenarios"][sid]
        run_id = result.get("worker_run_id")
        if type(run_id) is not int or run_id < 1:
            raise ValueError("Invalid worker run identifier")
        run = gh.api("GET", f"/actions/runs/{run_id}")
        jobs = gh.api("GET", f"/actions/runs/{run_id}/jobs?per_page=100")["jobs"]
        validate_worker_run(run, jobs, item, sid)
        outcome = gh.mutate(lambda state: apply_result(state, result))
        summary({"scenario": sid, "result": outcome, "worker_run_id": run_id})
        return
    if phase not in ("start", "checkpoint", "resume"):
        raise ValueError("Unsupported controller phase")
    sid = scenario_id(os.environ.get("PROBE_SCENARIO") or "probe-" + os.environ["GITHUB_RUN_ID"])
    if phase != "resume":
        gh.mutate(lambda state: prepare(state, sid, os.environ["GITHUB_SHA"]))
    ledger, _ = gh.read_state()
    item = ledger["scenarios"][sid]
    if phase == "checkpoint":
        summary({"scenario": sid, "result": "checkpoint_saved_without_dispatch"})
        return
    if item["phase"] == "completed" or item["dispatch"]["state"] == "submitted":
        summary({"scenario": sid, "result": "already_dispatched_or_completed"})
        return
    existing = gh.worker_run_for(sid)
    if existing is None:
        gh.dispatch(WORKER, {"scenario_id": sid})
    def mark_submitted(state):
        current = state["scenarios"][sid]
        current["dispatch"]["state"] = "submitted"
        if existing:
            current["dispatch"]["worker_run_id"] = existing["id"]
        if current["phase"] != "completed":
            current["phase"] = "running"
        return "submitted"
    gh.mutate(mark_submitted)
    summary({"scenario": sid, "result": "worker_dispatched", "existing_run_reused": bool(existing)})


def worker():
    gh = GitHub()
    sid = scenario_id(os.environ["PROBE_SCENARIO"])
    ledger, _ = gh.read_state()
    item = ledger["scenarios"][sid]
    if item["source_sha"] != os.environ["GITHUB_SHA"]:
        raise ValueError("Worker checkout differs from task source")
    if item["phase"] == "completed":
        raise ValueError("Task is already completed")
    result = result_for(sid, item, int(os.environ["GITHUB_RUN_ID"]))
    encoded = base64.b64encode(json.dumps(result).encode()).decode()
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf8") as file:
        file.write("result_b64=" + encoded + "\n")
    summary({"scenario": sid, "computed_value": result["value"], "worker_run_id": result["worker_run_id"]})


def publish():
    result = json.loads(base64.b64decode(os.environ["PROBE_RESULT_B64"]))
    gh = GitHub()
    sid = scenario_id(result["scenario_id"])
    stale = dict(result, goal_revision=0)
    for label, payload in (("original", result), ("duplicate", result), ("stale", stale)):
        gh.dispatch(CONTROLLER, {"phase": "ingest", "scenario_id": sid,
                                 "result_json": json.dumps(payload)})
        print("Dispatched controlled result fixture:", label)


if __name__ == "__main__":
    try:
        {"controller": controller, "worker": worker, "publish": publish}[sys.argv[1]]()
    except (APIError, ValueError, KeyError) as exc:
        print("Probe failed:", str(exc), file=sys.stderr)
        sys.exit(1)
