# Healthy pre-rollout diagnostics sample

Run window: March 6, 2026, 14:18 UTC

Operator summary:
- Parent diagnostics run opened normally from the on-call UI.
- The run page showed a nested investigation section with one child diagnostic step.
- The child diagnostic step stayed attached to the parent run and could be opened directly from the parent view.
- A failing probe still surfaced a useful run-level summary on the parent page, including what failed and where the operator should look next.

Observed sequence:
1. Parent diagnostics run starts from the incident page.
2. Parent run launches one deeper investigation step after initial checks complete.
3. Parent view updates to show the deeper investigation card inline.
4. One downstream probe fails.
5. Parent run summary changes from healthy to degraded and includes a readable failure reason.
6. Operator can still open the deeper investigation details from the same parent run page.

What on-call relied on before the rollout:
- Parent and deeper investigation activity appeared as one coherent debugging session.
- A failed deeper investigation step still left enough context on the parent run to understand what broke.
- Even when setup took longer than expected, the run page existed early enough that support could share the link while the investigation was still in progress.

Why this matters:
- The product behavior is not just whether diagnostics can execute.
- The operational value is whether one person can follow the run from launch, through deeper investigation, to failure context without reconstructing it from raw logs.
