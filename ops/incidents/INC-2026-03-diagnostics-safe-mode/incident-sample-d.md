# Broken sample D: follow-on work happened, but the parent run stayed too quiet

Run window: March 9, 2026, 14:02 UTC

Responder notes:
- Parent diagnostics run is still the place responders start from.
- Something in the deeper path clearly ran long enough to change downstream timing.
- The parent run never emitted a stable reduced-mode breadcrumb, summary, or fallback artifact that the operator could use.

Captured sequence:
- 14:02:03 parent run launched
- 14:02:11 initial diagnostics checks completed
- 14:02:19 downstream timing shifted again
- 14:02:21 parent run still showed no useful follow-on artifact

What the responder needed:
- one parent-facing note that said the deeper path was degraded, unavailable, or completed with reduced visibility
- enough durable context to keep the parent run as the canonical incident handle
