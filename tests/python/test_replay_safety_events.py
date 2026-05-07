"""
Test for scripts/replay_safety_events.py + transport/safety_replay.py
(P2-M5.A5).

Spawns a transient nats-server -js, provisions the safety_events stream,
publishes 5 well-formed SafetyEvents (which the replay should PASS) plus
1 deliberately-malformed event (replay should FAIL), then runs the
harness and checks the table.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import nats  # noqa: E402

from transport.init_streams import provision_all  # noqa: E402
from transport.proto_codec import pack_safety_event  # noqa: E402
from transport.safety_replay import evaluate  # noqa: E402

from transport.replay_safety_events import replay  # noqa: E402

NATS_SERVER = shutil.which("nats-server") or os.path.expanduser(
    "~/.local/bin/nats-server"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def transient_nats_server():
    if not Path(NATS_SERVER).is_file() or not os.access(NATS_SERVER, os.X_OK):
        pytest.skip(f"nats-server binary not found at {NATS_SERVER}")
    port = _free_port()
    store = tempfile.mkdtemp(prefix="natsjs_replay_")
    proc = subprocess.Popen(
        [NATS_SERVER, "-js", "-p", str(port), "-sd", store],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.terminate()
        proc.wait(timeout=3)
        pytest.fail("nats-server did not come up within 5 seconds")

    try:
        yield f"nats://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(store, ignore_errors=True)


def _make_event(v: float, d: float, v_w: float, trav: float, ts: float) -> bytes:
    """Compute the rule the supervisor WOULD produce for these inputs and
    publish that as the event's `rule` — so a faithful replay PASSes."""
    rule, scale = evaluate(v, d, v_w, trav)
    event = {
        "timestamp": ts,
        "rule": rule,
        "trigger_value": d,
        "vel_before": v,
        "vel_after": v * scale,
        "details": json.dumps({"v": v, "d": d, "v_w": v_w, "trav": trav}),
    }
    return pack_safety_event(event)


async def _publish_events(url: str, payloads: list[bytes]) -> None:
    nc = await nats.connect(url)
    try:
        js = nc.jetstream()
        for p in payloads:
            await js.publish("safety.events", p)
    finally:
        await nc.drain()


def test_replay_pass_fail_table(transient_nats_server):
    url = transient_nats_server
    asyncio.run(provision_all(url))

    # 5 well-formed scenarios spanning the rule set.
    well_formed = [
        _make_event(v=2.0, d=50.0, v_w=0.0, trav=1.0, ts=1.0),   # TTC >= 5s
        _make_event(v=2.0, d=2.0, v_w=0.0, trav=0.5, ts=2.0),    # TTC < 5s
        _make_event(v=3.0, d=1.5, v_w=0.0, trav=0.2, ts=3.0),    # TTC < 2s
        _make_event(v=2.0, d=0.5, v_w=0.0, trav=0.5, ts=4.0),    # d_worker < d_stop
        _make_event(v=1.0, d=20.0, v_w=-0.5, trav=1.0, ts=5.0),  # TTC >= 5s (worker receding)
    ]

    # 1 deliberately-malformed event: a real protobuf with an unknown rule
    # string. Replay should FAIL on schema validation.
    from transport.proto.safety_pb2 import SafetyEvent
    bad = SafetyEvent()
    bad.timestamp = 99.0
    bad.rule = "this_is_not_a_known_rule"
    bad.trigger_value = 1.0
    bad.vel_before = 1.0
    bad.vel_after = 1.0
    bad.details = "{}"
    payloads = well_formed + [bad.SerializeToString()]

    asyncio.run(_publish_events(url, payloads))

    rows = asyncio.run(replay(url, "safety_events", "safety.events", limit=100))

    statuses = [r.status for r in rows]
    assert statuses.count("PASS") == 5, f"expected 5 PASS rows, got {statuses}"
    assert statuses.count("FAIL") == 1, f"expected 1 FAIL row, got {statuses}"
    fail_row = next(r for r in rows if r.status == "FAIL")
    assert "unknown rule" in fail_row.reason


def test_safety_replay_known_vectors():
    """Hand-checked decision vectors — the C++/Python ports must agree.
    If C++ rules change, this test (and the port) must be updated."""
    # 1) safe at distance.
    rule, _ = evaluate(v_vehicle=2.0, d_worker=50.0, v_worker=0.0, trav_score=1.0)
    assert rule == "TTC >= 5s"

    # 2) E-stop when inside stopping distance.
    #    v=2, mu=trav_to_mu(0.5)=0.55, d_stop = 4/(2*0.55*9.81) + 0.4 ≈ 0.77 m.
    rule, scale = evaluate(v_vehicle=2.0, d_worker=0.5, v_worker=0.0, trav_score=0.5)
    assert rule == "d_worker < d_stop"
    assert scale == 0.0

    # 3) HARD_BRAKE when TTC < 2s.
    rule, _ = evaluate(v_vehicle=3.0, d_worker=1.5, v_worker=0.0, trav_score=0.2)
    # mu=0.4, d_stop = 9/(2*0.4*9.81) + 0.6 ≈ 1.747; v_rel=3; ttc≈-0.082 -> < 0 -> ESTOP.
    # Shows the ESTOP wins over hard-brake when d < d_stop.
    assert rule == "d_worker < d_stop"

    # 4) Receding worker -> safe even if close.
    rule, _ = evaluate(v_vehicle=1.0, d_worker=20.0, v_worker=-0.5, trav_score=1.0)
    assert rule == "TTC >= 5s"
