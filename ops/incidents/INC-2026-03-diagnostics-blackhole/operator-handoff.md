# Support handoff

Incident: INC-2026-03-diagnostics-blackhole

Operator notes:
- "Diagnostics still runs, but it is no longer useful when the incident gets complicated."
- "The top-level run opens, but I cannot always continue the investigation from the same place."
- "Some failed runs show a red state with no explanation I can use."
- "When launch fails early, I sometimes have nothing shareable to hand to the next responder."

Context carried into handoff:
- this does not look like a full diagnostics outage
- main run visibility still exists in some affected cases
- a rollout landed the night before
- a separate downstream deploy also happened in the same evening window

Unresolved questions at handoff time:
- whether the symptoms are connected or only overlap in timing
- whether the missing investigation continuity and weak failure summaries have the same cause
- whether the launch-time visibility problem belongs to the same incident or only surfaced in the same window
