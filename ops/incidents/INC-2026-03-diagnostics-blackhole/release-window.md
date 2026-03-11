# Rollout timeline

## March 8, 2026

18:05 UTC
- Platform rollout begins for the new diagnostics transport/orchestration bundle.

18:22 UTC
- Release notes mention cleanup to launch behavior and investigation execution handoffs.

18:47 UTC
- One internal smoke run is marked successful.
- The smoke run validates that the main diagnostics path still launches.
- It does not explicitly validate all degraded or early-failure visibility paths.

19:10 UTC
- Support opens the first internal note that diagnostics "still starts" but feels less useful during escalation.

20:03 UTC
- A separate downstream service deploy happens in the same general evening window.

## March 9, 2026

02:14 UTC
- On-call report: parent diagnostics runs open, but follow-on investigation is not consistently visible from the same run.

07:31 UTC
- On-call report: some failed diagnostics runs now end with thin or generic failure context on the main run page.

12:31 UTC
- On-call report: some setup-time failures produce no usable run link, making handoff difficult.
