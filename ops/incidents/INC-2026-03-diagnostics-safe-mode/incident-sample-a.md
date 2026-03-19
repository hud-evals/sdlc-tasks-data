# Broken sample A: no usable reduced-mode trail

Run window: March 9, 2026, 09:42 UTC

Responder notes:
- Parent diagnostics run opens.
- Initial checks finish and the run page stays green longer than expected.
- The operator expects either a deeper drill-down card or an explicit reduced-mode note on the parent run.
- Instead, the parent run gives no stable follow-on breadcrumb even though downstream timestamps suggest more work happened.

Captured timeline:
- 09:42:11 parent run launched from incident view
- 09:42:18 first diagnostics stage marked complete
- 09:42:27 no deeper card or reduced-mode note appeared on the parent run
- 09:42:31 downstream timestamps moved again without a usable parent-facing breadcrumb

What the responder still needed:
- a stable parent-run artifact showing whether deeper investigation was skipped, degraded, or continued elsewhere
- enough context to hand the incident off without leaving the main run page

What made this unusable:
- the parent run stayed alive
- the deeper path may still have done work
- but the responder had no durable trail on the parent run to follow or share
