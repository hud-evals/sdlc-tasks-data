# Broken sample A: detached child investigation

Run window: March 9, 2026, 09:42 UTC

Operator summary:
- Parent diagnostics run still launches and appears healthy at first glance.
- A deeper investigation step appears to execute somewhere in the system, but it is not attached back to the parent run in a usable way.
- On-call can tell that deeper work happened because downstream timestamps move, but the parent view never shows a coherent drill-down path.

Observed sequence:
1. Parent diagnostics run starts from the incident page.
2. Parent run performs initial checks and remains visible.
3. Parent run never shows the expected deeper investigation card.
4. Logs and downstream timestamps suggest the deeper step still executed.
5. Operator cannot navigate from the parent run to the missing investigation details from the same run view.

Confusing aspects during triage:
- This initially looks like a UI or navigation issue because the parent run exists.
- It also looks like a transport issue because some downstream evidence proves deeper work is happening.
- It is not immediately obvious whether the problem is "child work never happened" or "child work happened but became detached from the parent session."

What makes this a real on-call problem:
- Diagnostics technically still runs, so simple uptime checks look healthy.
- The missing drill-down path makes the diagnostics run operationally weak even when the parent run itself succeeds.
