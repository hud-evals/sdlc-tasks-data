# Broken sample C: no usable link on setup-time failure

Run window: March 9, 2026, 12:31 UTC

Responder notes:
- Launch fails very early.
- Console/session output shows the failure immediately.
- The responder cannot recover a usable run page link from the normal workflow.

Captured sequence:
- 12:31:02 diagnostics launch requested
- 12:31:04 setup failure surfaced to the responder
- 12:31:09 handoff attempt failed because no usable run page could be located

Operational impact:
- The responder can tell that launch failed.
- The responder cannot hand the failure to the next person in the same way they normally would.
- This turns a setup failure into a visibility problem as well as an execution problem.
