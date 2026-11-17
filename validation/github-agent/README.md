# GitHub-native agent infrastructure probe

This is an isolated infrastructure experiment, **not Argus, an LLM agent, or a modification to NRC's renderer**.

Scope:

- Source: `codex/github-agent-validation`.
- Durable test state: `codex-agent-probe-state`, `validation-state/ledger.json`.
- Static browser page: a separate Pages branch, with no credentials or product source.
- Controller and worker are different Workflow Runs.
- No model API keys, external services, or self-hosted runners are used.

The worker computes a fixed `6 * 7` task. A separate publisher delivers its result twice and a deliberately stale result once. The controller verifies the originating workflow, scenario, commit and successful execution job, accepts one result, deduplicates the second delivery and rejects the stale version.

`checkpoint` persists a task without dispatching it. `resume` in another Run loads that task from Git and continues. State remains accessible after all Runner processes exit.

## Local checks

```sh
python -m unittest discover -s validation/github-agent -p 'test_*.py' -v
```

## Observed validation (2026-11-17, Asia/Shanghai)

- The Pages origin `https://beyondtheapex.github.io/NRC/` served successfully, and a real browser could read the public GitHub REST API and durable ledger.
- The user authenticated on the Pages page and submitted scenario `probe-ui-1794879608260`. Controller run [43997317655](https://github.com/BeyondtheApex/NRC/actions/runs/43997317655), separate worker run [43997348892](https://github.com/BeyondtheApex/NRC/actions/runs/43997348892), and all three result-ingestion runs succeeded. The ledger records `completed`, result `42`, one duplicate delivery, one rejected stale goal revision, and exactly one completion event. Authentication and submission were user-operated; server-side execution and state were independently checked through `gh`.
- For `probe-recovery-001`, checkpoint run [43905461289](https://github.com/BeyondtheApex/NRC/actions/runs/43905461289) exited before dispatch. Later resume run [43996929373](https://github.com/BeyondtheApex/NRC/actions/runs/43996929373) loaded the Git state and dispatched separate worker [43996959344](https://github.com/BeyondtheApex/NRC/actions/runs/43996959344). All ingestion runs succeeded with the same expected terminal ledger checks.
- These live scenarios ran source commit `63ea7f31b565b67abd19d8ea86064c2aac64af16`. The subsequent scenario-binding patch adds a twelfth local protocol test; it is distinct from the source used in the browser test.
- The default branch `integration-sdk-paths` remained at `7d5b1357181e33c81b639297c780c07944c1cb16`; the draft PR changes only the probe workflows and `validation/github-agent/`.
- Browser requests to the Device Flow and OAuth token endpoints with an invalid placeholder Client ID were blocked by CORS. This does **not** establish the outcome of an enabled, valid App's complete Device Flow. That flow remains unverified. The successful login path here used a user-provided PAT held only in page memory.

## Limits of this evidence

Passing the probe validates these infrastructure behaviors only. It does not establish Argus checkpoint completeness, long-term autonomous research quality, secure execution of arbitrary untrusted code, exactly-once dispatch, or a complete OAuth login.

Branch-only dispatch succeeded in this repository after push registration, without merging the workflows into the default branch. Treat this as an observation of this probe, not a portability guarantee. Do not work around dispatch failures in other installations by silently modifying the default branch.

The controller's Git Contents API updates use version checks and bounded retries. A successful dispatch followed by a lost acknowledgement is reconciled against visible worker runs when resuming, but eventual-consistency races are not proven eliminated by this probe.

The task records an immutable source SHA, but this probe dispatches the workflow at a branch name. Advancing that branch while a task is pending can cause a source-mismatch rejection. The live scenarios were completed before advancing the source branch. A production implementation must version its execution runtime and define checkpoint migration/retry behavior across upgrades. This probe also does not recover an already-submitted worker that fails or disappears.

The worker's scenario title check binds its successful execution evidence to this fixture. It does not constitute a general trust boundary for arbitrary worker output. No external issue content, model-generated code, self-upgrade, or hostile worker was tested.

## Cleanup

After review, remove the probe workflows/branch, state branch and Pages test branch, and disable the test Pages site if it is no longer required. Preserve the run links and final ledger as evidence first. No cleanup is automatic.
