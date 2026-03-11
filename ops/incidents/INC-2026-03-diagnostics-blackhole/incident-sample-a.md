# Broken sample A: detached child investigation

Run window: March 9, 2026, 09:42 UTC

Responder notes:
- Parent diagnostics run opens.
- Initial checks finish and the run page stays green longer than expected.
- Operator expects a deeper drill-down card but never sees one on the parent page.
- Separate timestamps in downstream output suggest another step may have executed anyway.

Captured timeline:
- 09:42:11 parent run launched from incident view
- 09:42:18 first diagnostics stage marked complete
- 09:42:27 no additional investigation card appeared on the same page
- 09:42:31 downstream timestamps moved again without a visible drill-down from the parent run

Open questions from handoff:
- Did the deeper step never start?
- Did it start somewhere else and lose continuity with the parent run?
- Is this a view problem, an execution problem, or something in between?
