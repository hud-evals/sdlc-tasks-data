# Support handoff

Incident: INC-2026-03-diagnostics-safe-mode

Operator notes:
- "Diagnostics still launches often enough to look alive, but when the incident gets messy the run stops being something I can trust or hand off."
- "If the deeper path cannot stay attached cleanly, I would rather get a clear reduced-mode trail on the parent run than silence."
- "A broad shutdown would be expensive tonight. We still need diagnostics to be usable enough for responders before a full repair lands."
- "Early launch failures are also part of the pain because they leave me with nothing durable to hand to the next person."

Context carried into handoff:
- this does not look like a full diagnostics outage
- the parent run often still exists
- the rollout landed the night before
- a full restoration may take longer than the current incident window allows

What the next responder needs answered:
- what reduced or safe behavior would still leave one dependable incident handle
- what must stay visible on the parent run even when deeper investigation cannot behave normally
- how to preserve handoff quality without pretending diagnostics is fully healthy
