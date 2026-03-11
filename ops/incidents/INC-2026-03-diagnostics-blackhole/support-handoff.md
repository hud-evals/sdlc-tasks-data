# Support handoff

Incident: INC-2026-03-diagnostics-blackhole

What operators are saying:
- "Diagnostics still runs, but it is no longer useful when the incident gets complicated."
- "The top-level run opens, but the deeper investigation path is sometimes missing from the same view."
- "Some failed runs show a red state with no explanation I can use."
- "When launch fails early, I sometimes have nothing shareable to hand to the next responder."

What we know so far:
- This does not look like a full diagnostics outage.
- The parent run path still exists in many broken examples.
- The problem is concentrated in debuggability and operational visibility, not basic launch success.
- There was a rollout the night before involving diagnostics execution, deeper investigation handling, and early launch behavior.

What remains unclear:
- whether the problem is one regression family or several unrelated regressions
- whether the missing deeper investigation view and the blank failure context share the same cause
- whether the no-link setup failures come from the same rollout or from a separate setup bug
