"""
Test for transport/init_streams.py (P2-M5.A4).

Spawns a transient nats-server with JetStream on a temp store dir + a
non-default port, runs the provisioner twice, and asserts:
  1. First run reports "created".
  2. Second run reports "no_change" (idempotency).
  3. Stream config seen by `js.stream_info` matches the StreamSpec.

Skipped automatically if `nats-server` is not on PATH.
"""

from __future__ import annotations

import asyncio
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

from transport.init_streams import provision_all, PROJECT_STREAMS  # noqa: E402

NATS_SERVER = shutil.which("nats-server") or os.path.expanduser(
    "~/.local/bin/nats-server"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def transient_nats_server():
    """Spin up nats-server -js on a free port + temp store dir."""
    if not Path(NATS_SERVER).is_file() or not os.access(NATS_SERVER, os.X_OK):
        pytest.skip(
            f"nats-server binary not found at {NATS_SERVER}. "
            "Install with: download from nats.io and place in ~/.local/bin/",
        )
    port = _free_port()
    store = tempfile.mkdtemp(prefix="natsjs_test_")
    proc = subprocess.Popen(
        [NATS_SERVER, "-js", "-p", str(port), "-sd", store],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for the broker to be ready (poll TCP connect).
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


def test_provisioner_is_idempotent(transient_nats_server):
    url = transient_nats_server

    # Run 1: should create.
    results1 = asyncio.run(provision_all(url))
    actions1 = {r.name: r.action for r in results1}
    for spec in PROJECT_STREAMS:
        assert actions1[spec.name] == "created", (
            f"first run on a fresh broker should create stream {spec.name}; "
            f"got {actions1[spec.name]}"
        )

    # Run 2: should be a no-op.
    results2 = asyncio.run(provision_all(url))
    actions2 = {r.name: r.action for r in results2}
    for spec in PROJECT_STREAMS:
        assert actions2[spec.name] == "no_change", (
            f"second run should report no_change for stream {spec.name}; "
            f"got {actions2[spec.name]}"
        )


def test_stream_config_after_provision(transient_nats_server):
    """After provisioning, the stream config the broker reports should
    match the StreamSpec fields we set."""
    import nats

    url = transient_nats_server
    asyncio.run(provision_all(url))

    async def inspect():
        nc = await nats.connect(url)
        try:
            js = nc.jetstream()
            for spec in PROJECT_STREAMS:
                info = await js.stream_info(spec.name)
                assert info.config.name == spec.name
                assert sorted(info.config.subjects or []) == sorted(spec.subjects)
                assert (info.config.max_bytes or -1) == spec.max_bytes
                assert info.config.retention == spec.retention
                assert info.config.storage == spec.storage
        finally:
            await nc.drain()

    asyncio.run(inspect())
