"""
replay_safety_events.py — thin CLI shim over transport/replay_safety_events.py.

Logic lives in `transport.replay_safety_events` so the test
(`tests/python/test_replay_safety_events.py`) can import the same
functions without `scripts/` needing to be a package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transport.replay_safety_events import main

if __name__ == "__main__":
    sys.exit(main())
