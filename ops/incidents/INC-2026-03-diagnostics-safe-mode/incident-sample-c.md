# Broken sample C: early failure with no shareable incident handle

Run window: March 9, 2026, 12:31 UTC

Responder notes:
- Launch fails very early.
- Console/session output shows the failure immediately.
- The responder still cannot recover a usable run page link from the normal workflow.

Captured sequence:
- 12:31:02 diagnostics launch requested
- 12:31:04 setup failure surfaced to the responder
- 12:31:09 handoff attempt failed because no usable run page could be located

What the responder could have worked with:
- a durable run handle, even if the run entered a reduced or failed state immediately
- a stable link or incident artifact that could be handed to the next person

Why this is part of the same incident family:
- launch-time failure is not the only problem
- the larger issue is that responders lose a dependable incident handle exactly when the path degrades
