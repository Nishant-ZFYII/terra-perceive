"""
replay_safety_events.py — JetStream audit-trail replay harness (P2-M5.A5).

Reads every persisted SafetyEvent from the `safety_events` durable stream
(provisioned by transport/init_streams.py), re-evaluates each event
through `transport/safety_replay.py` (a faithful port of the C++
SafetySupervisor rules), and prints a pass/fail table:

  * PASS — rule that fired at recording time matches the rule the
           current logic produces for the same inputs.
  * FAIL — current logic disagrees, OR the event is structurally
           malformed (missing rule, NaN velocities, undecodable details, ...).
  * SKIP — event lacks the `details` JSON encoding the inputs needed
           for replay; we can validate schema but not the rule.

The producer (a future C++ supervisor publish or `proto_codec.pack_safety_event`
caller) is expected to encode the inputs that drove the rule into the
`details` field as JSON: `{"v": v, "d": d, "v_w": v_w, "trav": trav}`.
The test publishes events in that shape.

Usage:
  python scripts/replay_safety_events.py --nats-url nats://localhost:4222 [--limit 100]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import nats  # noqa: E402
from nats.errors import NoServersError, TimeoutError as NatsTimeoutError  # noqa: E402

from transport.proto_codec import unpack_safety_event  # noqa: E402
from transport.safety_replay import evaluate  # noqa: E402

log = logging.getLogger("replay_safety_events")

DEFAULT_STREAM = "safety_events"
DEFAULT_SUBJECT = "safety.events"
DEFAULT_NATS_URL = "nats://localhost:4222"
KNOWN_RULES = {"d_worker < d_stop", "TTC < 2s", "TTC < 5s", "TTC >= 5s"}


@dataclass
class ReplayRow:
    seq: int
    timestamp: float
    rule_recorded: str
    rule_replayed: Optional[str]
    status: str   # "PASS" | "FAIL" | "SKIP"
    reason: str = ""


def _try_parse_details(details: str) -> Optional[dict]:
    if not details:
        return None
    try:
        d = json.loads(details)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    needed = {"v", "d", "v_w", "trav"}
    if not needed.issubset(d.keys()):
        return None
    return d


def _validate_schema(event: dict) -> Optional[str]:
    """Return None if schema looks ok, else a string describing the problem."""
    rule = event.get("rule") or ""
    if not rule:
        return "missing or empty rule"
    if rule not in KNOWN_RULES:
        return f"unknown rule: {rule!r}"
    for k in ("vel_before", "vel_after", "trigger_value"):
        v = event.get(k)
        if v is None or not math.isfinite(float(v)):
            return f"non-finite {k}={v!r}"
    return None


def replay_one(seq: int, payload: bytes) -> ReplayRow:
    try:
        event = unpack_safety_event(payload)
    except Exception as e:
        return ReplayRow(seq, 0.0, "?", None, "FAIL", f"deserialize: {e}")

    schema_err = _validate_schema(event)
    if schema_err is not None:
        return ReplayRow(
            seq, event.get("timestamp", 0.0), event.get("rule") or "?",
            None, "FAIL", schema_err,
        )

    details = _try_parse_details(event.get("details") or "")
    if details is None:
        return ReplayRow(
            seq, event["timestamp"], event["rule"], None,
            "SKIP", "details lacks {v, d, v_w, trav}",
        )

    rule_replayed, _scale = evaluate(
        v_vehicle=float(details["v"]),
        d_worker=float(details["d"]),
        v_worker=float(details["v_w"]),
        trav_score=float(details["trav"]),
    )
    if rule_replayed == event["rule"]:
        return ReplayRow(seq, event["timestamp"], event["rule"], rule_replayed, "PASS")
    return ReplayRow(
        seq, event["timestamp"], event["rule"], rule_replayed,
        "FAIL", "current logic produces a different rule",
    )


async def replay(nats_url: str, stream: str, subject: str, limit: int) -> list[ReplayRow]:
    nc = await nats.connect(nats_url, name="terra_perceive_replay_harness")
    rows: list[ReplayRow] = []
    try:
        js = nc.jetstream()
        sub = await js.pull_subscribe(subject, durable="replay_harness", stream=stream)
        seq = 0
        while seq < limit:
            try:
                msgs = await sub.fetch(min(50, limit - seq), timeout=1.0)
            except NatsTimeoutError:
                break
            if not msgs:
                break
            for m in msgs:
                seq += 1
                rows.append(replay_one(seq, m.data))
                await m.ack()
    finally:
        await nc.drain()
    return rows


def print_table(rows: list[ReplayRow]) -> None:
    print(f"{'#':>4}  {'status':<6}  {'rule_recorded':<20}  {'rule_replayed':<20}  reason")
    print("-" * 90)
    for r in rows:
        rep = r.rule_replayed if r.rule_replayed is not None else "—"
        print(f"{r.seq:>4}  {r.status:<6}  {r.rule_recorded:<20}  {rep:<20}  {r.reason}")
    n_pass = sum(1 for r in rows if r.status == "PASS")
    n_fail = sum(1 for r in rows if r.status == "FAIL")
    n_skip = sum(1 for r in rows if r.status == "SKIP")
    print("-" * 90)
    print(f"total={len(rows)}  PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--nats-url", default=DEFAULT_NATS_URL)
    p.add_argument("--stream", default=DEFAULT_STREAM)
    p.add_argument("--subject", default=DEFAULT_SUBJECT)
    p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args()

    try:
        rows = asyncio.run(
            replay(args.nats_url, args.stream, args.subject, args.limit),
        )
    except NoServersError as e:
        log.error("could not connect to %s — %s", args.nats_url, e)
        return 2
    print_table(rows)
    return 1 if any(r.status == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
