# Broken sample B: degraded state with no actionable handoff

Run window: March 9, 2026, 11:07 UTC

Responder notes:
- Parent run exists and changes state from healthy to degraded.
- A follow-up investigation step appears to have gone wrong.
- The visible run summary is too thin to route correctly.
- An operator would have accepted a reduced-mode explanation if it had still said what failed and what to do next.

Captured sequence:
- 11:07:04 parent run launched
- 11:07:16 follow-up investigation path started
- 11:07:24 parent run changed state
- 11:07:24 visible summary remained generic

What the responder could not answer from the run page:
- Did the deeper path fail, get skipped, or get cut over to reduced mode?
- Was the failure in setup, execution, or a downstream dependency?
- Was there any reliable next step other than abandoning the run page and searching elsewhere?

Operationally acceptable outcome:
- the system did not need full behavior
- it did need an explicit, durable, operator-usable summary on the parent run
