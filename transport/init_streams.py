"""
init_streams.py — idempotent JetStream provisioning for P2-M5.A4.

Declares the durable streams the project needs:
  safety_events  — subject filter `safety.events`, retention forever,
                   max_bytes ~1 GB. Stores every SafetyEvent the
                   safety supervisor publishes for after-the-fact replay.

Idempotent: safe to run repeatedly. If the stream already exists with a
matching config the script reports "no change" and exits 0; if config
differs the script updates the stream in place and reports "updated"; if
the stream is missing the script creates it.

Why we provision in code instead of `nats stream add` CLI:
  - The `nats` CLI is not installed in the conda env; nats-py is.
  - Keeping the schema in Python means the test (`test_init_streams.py`)
    can import the same desired-config and compare against what the
    broker reports — single source of truth.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import nats
from nats.errors import NoServersError
from nats.js import JetStreamContext
from nats.js.api import StreamConfig, RetentionPolicy, StorageType
from nats.js.errors import NotFoundError

log = logging.getLogger("init_streams")

DEFAULT_NATS_URL = "nats://localhost:4222"

# ~1 GB — generous for SafetyEvent records (each ~200 bytes serialized).
DEFAULT_MAX_BYTES = 1 * 1024 * 1024 * 1024


@dataclass
class StreamSpec:
    name: str
    subjects: list[str]
    max_bytes: int = DEFAULT_MAX_BYTES
    retention: RetentionPolicy = RetentionPolicy.LIMITS
    storage: StorageType = StorageType.FILE
    max_age_ns: int = 0  # 0 = forever for LIMITS retention.

    def to_stream_config(self) -> StreamConfig:
        return StreamConfig(
            name=self.name,
            subjects=list(self.subjects),
            retention=self.retention,
            max_bytes=self.max_bytes,
            storage=self.storage,
            max_age=self.max_age_ns,
        )


# The single project-wide stream registry. Add new entries here when a new
# durable subject is introduced (e.g. telemetry.events would go here too).
PROJECT_STREAMS: list[StreamSpec] = [
    StreamSpec(name="safety_events", subjects=["safety.events"]),
]


@dataclass
class ProvisionResult:
    name: str
    action: str            # "created" | "updated" | "no_change"
    config_after: dict = field(default_factory=dict)


def _config_matches(existing: StreamConfig, desired: StreamConfig) -> bool:
    """Compare the fields that actually matter for our use case. Other
    fields (replicas, mirror, sources, ...) we don't set so we don't compare."""
    return (
        sorted(existing.subjects or []) == sorted(desired.subjects or [])
        and existing.retention == desired.retention
        and (existing.max_bytes or -1) == (desired.max_bytes or -1)
        and existing.storage == desired.storage
        and (existing.max_age or 0) == (desired.max_age or 0)
    )


async def provision_one(js: JetStreamContext, spec: StreamSpec) -> ProvisionResult:
    desired = spec.to_stream_config()
    try:
        info = await js.stream_info(spec.name)
    except NotFoundError:
        await js.add_stream(config=desired)
        log.info("created stream %s", spec.name)
        return ProvisionResult(spec.name, "created")

    if _config_matches(info.config, desired):
        log.info("stream %s already matches desired config", spec.name)
        return ProvisionResult(spec.name, "no_change")

    await js.update_stream(config=desired)
    log.info("updated stream %s to match desired config", spec.name)
    return ProvisionResult(spec.name, "updated")


async def provision_all(nats_url: str = DEFAULT_NATS_URL) -> list[ProvisionResult]:
    nc = await nats.connect(nats_url, name="terra_perceive_init_streams")
    try:
        js = nc.jetstream()
        results = []
        for spec in PROJECT_STREAMS:
            results.append(await provision_one(js, spec))
        return results
    finally:
        await nc.drain()


async def _run(nats_url: str) -> int:
    try:
        results = await provision_all(nats_url)
    except NoServersError as e:
        log.error("could not connect to %s — is nats-server running?", nats_url)
        log.error("Start one with:  nats-server -js  (jetstream is required)")
        log.error("Or:              docker compose -f docker/docker-compose.yml up nats")
        log.error("Underlying:      %s", e)
        return 2

    for r in results:
        print(f"{r.name}: {r.action}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--nats-url", default=DEFAULT_NATS_URL)
    args = p.parse_args()
    return asyncio.run(_run(args.nats_url))


if __name__ == "__main__":
    sys.exit(main())
