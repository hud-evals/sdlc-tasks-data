# Broken sample B: blank failure context

Run window: March 9, 2026, 11:07 UTC

Responder notes:
- Parent run exists and changes state from healthy to degraded.
- A follow-up investigation step appears to have failed.
- The visible failure summary on the main run is too thin to act on.

Captured sequence:
- 11:07:04 parent run launched
- 11:07:16 secondary investigation path started
- 11:07:24 parent run changed state
- 11:07:24 visible summary remained generic

What the responder could not answer from the run page:
- Was the failure in setup, execution, or downstream dependency handling?
- Was the visible failure summary missing because nothing was recorded, or because it was not carried back to the main run?

What changed operationally:
- Before this incident family, the main run usually carried enough detail to decide whether to page the application owner or continue platform triage.
- In this sample, the operator had to leave the run page and search elsewhere immediately.
