"""
frame_io.py — 4-byte big-endian length-prefixed stream framing.

Wire format mirrors the C++ helpers in `ros2_nodes/tracker_node.cpp`
(`readLengthPrefixedFrame` / `writeLengthPrefixedFrame`):

    +-----------+-----------+
    | 4 bytes   | N bytes   |
    | length BE | payload   |
    +-----------+-----------+

Both ends use the same Python implementation by importing from this module
so the contract stays in lock-step. EOF on the read side is reported as
None (not an exception) to make the consumer's loop trivial.
"""

from __future__ import annotations

import struct
from typing import BinaryIO, Optional


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def read_frame(stream: BinaryIO) -> Optional[bytes]:
    """Return the next payload, or None if the stream is at EOF."""
    hdr = stream.read(4)
    if len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    payload = b""
    while len(payload) < n:
        chunk = stream.read(n - len(payload))
        if not chunk:
            return None
        payload += chunk
    return payload


async def aread_frame(stream) -> Optional[bytes]:
    """Async variant for asyncio StreamReader-style streams."""
    hdr = await stream.readexactly(4) if hasattr(stream, "readexactly") else None
    if hdr is None or len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    return await stream.readexactly(n)
