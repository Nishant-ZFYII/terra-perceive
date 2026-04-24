"""
nats_adapter.py — async NATS pub/sub adapter for the perception transport layer.

P2-M3 bootstrap scope:
  * Connect / disconnect to a NATS server.
  * Publish raw bytes (typically a serialized protobuf) to a subject.
  * Subscribe to a subject with an async handler.
  * Log every publish/receive so the demo is auditable from the terminal.

Out of scope here (deferred to M5+):
  * JetStream durable streams (only ephemeral pub/sub).
  * Flow control under load (no backpressure handling).
  * Message-level schema dispatch (callers serialize/deserialize themselves).

Subjects this project standardizes on (PDF Part 3 §3.2 addendum):
    perception.traversability.grid     — BEVGrid
    perception.objects.detections      — DetectionList         (M5)
    perception.objects.tracks          — TrackList             (M5)
    safety.cmd_vel.safe                — guarded velocity      (M7)
    telemetry.metrics                  — scalar metrics        (M7)
    telemetry.events                   — discrete events       (M7)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import nats
from nats.aio.client import Client as NATSClient

log = logging.getLogger("nats_adapter")

# Type alias: an async function that takes the raw payload bytes and returns nothing.
Handler = Callable[[bytes], Awaitable[None]]


class Adapter:
    """Thin async wrapper around nats-py."""

    def __init__(self, url: str = "nats://localhost:4222") -> None:
        self.url = url
        self._nc: Optional[NATSClient] = None
        self._n_pub = 0
        self._n_recv = 0

    async def connect(self) -> None:
        self._nc = await nats.connect(self.url, name="terra_perceive_adapter")
        log.info("connected to %s", self.url)

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
            log.info("closed (published=%d  received=%d)", self._n_pub, self._n_recv)
            self._nc = None

    async def publish(self, subject: str, payload: bytes) -> None:
        if self._nc is None:
            raise RuntimeError("publish() called before connect()")
        await self._nc.publish(subject, payload)
        self._n_pub += 1
        log.info("PUB %s  (%d bytes)  total=%d", subject, len(payload), self._n_pub)

    async def subscribe(self, subject: str, handler: Handler) -> None:
        if self._nc is None:
            raise RuntimeError("subscribe() called before connect()")

        async def _wrapped(msg) -> None:
            self._n_recv += 1
            log.info(
                "RECV %s  (%d bytes)  total=%d",
                msg.subject, len(msg.data), self._n_recv,
            )
            await handler(msg.data)

        await self._nc.subscribe(subject, cb=_wrapped)
        log.info("SUB %s", subject)

    @property
    def stats(self) -> dict:
        return {"published": self._n_pub, "received": self._n_recv}


# --------------------------------------------------------------------------- #
# Tiny self-test: run directly to verify the adapter against a local nats-server.
#   $ nats-server -js &
#   $ python -m transport.nats_adapter
# --------------------------------------------------------------------------- #
async def _selftest() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    received = asyncio.Event()

    pub = Adapter()
    sub = Adapter()
    await pub.connect()
    await sub.connect()

    async def handler(data: bytes) -> None:
        log.info("handler got: %r", data)
        received.set()

    await sub.subscribe("perception.test.echo", handler)
    await asyncio.sleep(0.05)  # let subscription register
    await pub.publish("perception.test.echo", b"hello terra")

    try:
        await asyncio.wait_for(received.wait(), timeout=2.0)
        log.info("self-test OK")
    except asyncio.TimeoutError:
        log.error("self-test TIMEOUT — is nats-server running on :4222?")
    finally:
        await pub.close()
        await sub.close()


if __name__ == "__main__":
    asyncio.run(_selftest())
