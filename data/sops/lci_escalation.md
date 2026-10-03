# LCI Escalation — Tiered Process (Email-Only)

**Category code:** ESC-TIER
**Note:** LCI uses EMAIL escalation only — Tier 1, Tier 2, Tier 3. There is
no watcher mechanism in LCI (watchers are SPS-only).

## Pre-step — OnCall lookup
Before any tier escalation, look up the current OnCall resolver for the queue
so the escalation reaches the right person on shift.

## Tier 1
First line. The Tier 1 OnCall attempts resolution within SLA. If Tier 1 can
resolve, no escalation is needed.

## Tier 2
If Tier 1 OnCall cannot resolve within SLA, escalate to Tier 2 via email to
the regional LCI queue. The email MUST include:
- Case ID
- Case age
- Tier 1 steps already taken
Tier 2 owns routing to the correct resolver group.

## Tier 3
Reserved for the most complex or policy-sensitive cases that Tier 2 cannot
resolve. Tier 3 handoffs are routed to the senior resolver and require a clear
summary of Tier 1 and Tier 2 actions.

## Audit guidance
- Record which tier owns the case under the escalation column.
- Dual-channel notification (Slack DM + activity feed) is used for Tier 3
  handoffs for safety.
