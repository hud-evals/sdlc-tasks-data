# Broken sample C: no usable link on setup-time failure

Run window: March 9, 2026, 12:31 UTC

Operator summary:
- The run fails during very early setup.
- Support sees a launch failure in the console/session output.
- There is no usable diagnostics run link to share with the next responder.

Observed sequence:
1. Operator starts diagnostics from the incident page.
2. Setup fails before the normal investigation flow becomes visible.
3. Console output shows the launch error immediately.
4. The team cannot recover a usable run page link from the normal operator workflow.

Why this is misleading:
- It initially looks like a pure setup or connection problem.
- But the real operational complaint is not only that setup failed.
- The larger problem is that on-call loses the ability to inspect, share, and correlate the failed run at the moment it is most needed.

Why this matters:
- Even a failed launch is normally still useful if the run becomes visible early enough to inspect later.
- After the rollout, some setup-time failures appear to die before that visibility is established.
