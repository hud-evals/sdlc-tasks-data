# Broken sample B: blank failure context

Run window: March 9, 2026, 11:07 UTC

Operator summary:
- Parent diagnostics run exists.
- A deeper investigation path fails.
- The parent run ends in a degraded or failed state but carries almost no usable explanation.

Observed sequence:
1. Parent diagnostics run launches successfully.
2. A deeper investigation path is triggered after the initial checks.
3. That deeper path fails.
4. Parent run summary changes state but the visible failure context is generic and not actionable.
5. On-call cannot tell whether the actual failure was transport, orchestration, dependency setup, or application behavior without leaving the run page and searching elsewhere.

What support reported:
- Before the rollout, the parent run summary usually contained enough detail to decide whether to wake the application owner or continue platform triage.
- After the rollout, some failed runs end with a generic failure shell and no useful reason attached to the main run.

Competing hypotheses this creates:
- maybe the failure is happening before any useful context is recorded
- maybe the deeper investigation result is failing but the parent run is not absorbing that failure correctly
- maybe the diagnostics transport changed how failure details are surfaced

What this rules out:
- This is not just "diagnostics never ran"
- Parent and deeper work both appear to happen, but the failure context is no longer preserved where on-call expects it
