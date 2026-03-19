# Rollout timeline

## March 8, 2026

18:05 UTC
- Platform rollout begins for the diagnostics transport / orchestration bundle.

18:22 UTC
- Release notes mention cleanup to launch behavior and investigation execution handoffs.

18:47 UTC
- One internal smoke run is marked successful.
- The smoke run validates that the main diagnostics path still launches.
- It does not explicitly validate reduced-mode behavior, early-failure handoff, or degraded nested investigation trails.

19:10 UTC
- Support opens the first internal note that diagnostics still starts but stops being dependable once the run gets beyond the initial checks.

20:03 UTC
- A separate downstream service deploy happens in the same general evening window.

## March 9, 2026

02:14 UTC
- On-call report: parent diagnostics runs open, but the same run no longer carries a usable deeper trail.

07:31 UTC
- On-call report: some degraded runs now end with summaries too thin to route or hand off.

12:31 UTC
- On-call report: some early launch failures produce no durable run handle.

## Operational constraint

- Enterprise escalations continue tonight.
- Full restoration may not land inside the incident window.
- The immediate ask is to preserve a responder-usable reduced mode rather than either broad shutdown or an overconfident partial fix.
