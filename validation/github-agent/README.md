# GitHub-native agent infrastructure probe

This is an isolated infrastructure experiment, **not Argus, an LLM agent, or a modification to NRC's renderer**.

Scope:

- Source: `codex/github-agent-validation`.
- Durable test state: `codex-agent-probe-state`, `validation-state/ledger.json`.
- Static browser page: a separate Pages branch, with no credentials or product source.
- Controller and worker are different Workflow Runs.
- No model API keys, external services, or self-hosted runners are used.

The worker computes a fixed `6 * 7` task. A separate publisher delivers its result twice and a deliberately stale result once. The controller verifies the originating workflow, commit and successful execution job, accepts one result, deduplicates the second delivery and rejects the stale version.

`checkpoint` persists a task without dispatching it. `resume` in another Run loads that task from Git and continues. State remains accessible after all Runner processes exit.

## Local checks

```sh
python -m unittest discover -s validation/github-agent -p 'test_*.py' -v
```

## Limits of this evidence

Passing the probe validates these infrastructure behaviors only. It does not establish Argus checkpoint completeness, long-term autonomous research quality, secure execution of arbitrary untrusted code, exactly-once dispatch, or a complete OAuth login.

The branch-only availability of `workflow_dispatch` is explicitly being tested. GitHub documentation describes a default-branch requirement; if GitHub refuses branch-only dispatch, this probe should fail visibly. Do not work around that failure by silently modifying the default branch.

The controller's Git Contents API updates use version checks and bounded retries. A successful dispatch followed by a lost acknowledgement is reconciled against visible worker runs when resuming, but eventual-consistency races are not proven eliminated by this probe.

## Cleanup

After review, remove the probe workflows/branch, state branch and Pages test branch, and disable the test Pages site if it is no longer required. Preserve the run links and final ledger as evidence first. No cleanup is automatic.
