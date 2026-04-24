# P2-M3: NATS Transport Bootstrap

This is the smoke-test walkthrough for the M3 transport layer. It reproduces the live BEV dashboard demo: an offline accumulator run is replayed through a real NATS server, deserialized in a separate process, and rendered as a live matplotlib heatmap.

The maps in the dashboard are the same ones produced by `accumulator_runner` and shown elsewhere in the M3 results. Nothing about the underlying SLAM, ground segmentation, or grid update is changed by this layer — NATS is purely the wire between producer and consumer.

## Architecture

```
┌─────────────────────────┐                    ┌─────────────────────────────┐
│  publish_grid_stream.py │  serializes        │   nats-server               │
│                         │  BEVGrid proto     │   (or docker compose nats)  │
│  - reads snapshots/     │ ─────────────────► │                             │
│  - reads pose_sigma.csv │                    │   subject:                  │
│  - paces at --hz        │                    │   perception.traversability.grid
│                         │                    │                             │
└─────────────────────────┘                    └─────────────┬───────────────┘
                                                             │
                                                             │ async subscribe
                                                             ▼
                                             ┌─────────────────────────────┐
                                             │  dashboard_subscriber.py    │
                                             │                             │
                                             │  - deserializes BEVGrid     │
                                             │  - matplotlib live heatmap  │
                                             │  - shows frame_id,          │
                                             │    pose_sigma, latency      │
                                             └─────────────────────────────┘
```

Three terminals: NATS server, publisher, subscriber. Nothing else.

## What's deferred (M5 / M7)

- Live publishing from inside `accumulator_runner.cpp` (C++ NATS client). The stub at `src/accumulator_runner.cpp:472` is wired but not implemented; `--publish-nats` parses, the publish call is a no-op. M5 will fill this in alongside the C++ tracker publisher.
- JetStream durable streams. The broker runs with `--jetstream` enabled, but no streams are declared yet. Telemetry / safety subjects are the natural first JetStream candidates.
- Backpressure / flow control. Our 5 Hz BEVGrid stream at ~4 MB/message saturates a localhost socket fine; this would need attention before networked deployment.
- Authentication / TLS. Demo runs against an open `nats-server`. Production would need user-credentials and TLS certs.

## Why NATS, not gRPC or Kafka

| | gRPC | Kafka | **NATS** |
|---|---|---|---|
| Pattern | 1:1 RPC | Durable distributed log | Lightweight pub/sub |
| Producer blocks if consumer slow | Yes (HTTP/2 flow control) | No | No (default at-most-once) |
| Operational footprint | Library only | JVM + ZK/KRaft, ~GB RAM | Single ~15 MB binary |
| Runs on the robot | Yes (it's a library) | No (cluster off-board) | Yes |
| Hierarchical subjects | No | No (flat topics) | Yes (`perception.>`) |
| Replay / time travel | None | Native | Optional via JetStream |

For a 10 Hz BEV stream with N independent consumers (planner, dashboard, logger, future ML training pipeline) running on edge hardware, NATS is the right shape. gRPC would force 1:1 service interfaces; Kafka would add infra weight that the on-robot compute box can't justify.

We will use Kafka — but for the offline data lake (replay → S3 → training), not for the inference hot loop. We will use gRPC — but for synchronous service calls (e.g. map-tile lookup), not for streams.

## Schema discipline

All messages share a common header — see `transport/proto/perception.proto:11-16`:

```proto
message Header {
    double timestamp     = 1;   // wall-clock at producer
    string frame_id      = 2;   // unique frame label, e.g. "frame_002847"
    uint32 schema_version = 3;  // bump on breaking change
    string source        = 4;   // producer module name
}
```

`BEVGrid` carries the header plus `pose_sigma_at_snapshot` (line 25), populated by `publish_grid_stream.py` from each run's `pose_sigma.csv`. This is the live downstream signal of SLAM uncertainty that Ablation E (true marginals via `computeMarginalsG2O`) actually validates — it's not just a header field, it's a real number that consumers can use to gate their own decisions on map confidence.

## Run the demo

### 0. Generate Python protobuf bindings (once)

```bash
bash transport/generate_protos.sh
# expected: Generated:
#   transport/proto/perception_pb2.py  (and others)
```

If `protoc` isn't installed:

```bash
sudo apt install protobuf-compiler
```

### 1. Start the broker (terminal 1)

Either via Docker:

```bash
docker compose -f docker/docker-compose.yml up nats
```

…or natively if you have nats-server installed:

```bash
nats-server -js
# expected: [INF] Server is ready, listening on :4222
```

### 2. Start the BEV publisher (terminal 2)

```bash
python scripts/publish_grid_stream.py \
    --run results/m3/slam_ema_covg2o_full/ \
    --hz 5 \
    --loop
```

### 2b. Start the camera publisher (terminal 3) — second producer

```bash
python scripts/publish_camera_stream.py \
    --camera-dir data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node \
    --hz 5 \
    --loop
```

Two **independent** producer processes write to two different NATS subjects.
Neither knows the other exists — that's the pub/sub abstraction working.

`--loop` makes them cycle their sequences indefinitely so you can leave them
running while you set up the recording.

Expected output:

```
INFO  loaded 57 snapshots from results/m3/slam_ema_covg2o_full/snapshots
INFO  publishing on subject perception.traversability.grid at 5.00 Hz
INFO  connected to nats://localhost:4222
INFO  PUB perception.traversability.grid  (~4128127 bytes)  total=1
INFO  PUB perception.traversability.grid  (~4128127 bytes)  total=2
...
```

### 3. Start the dashboard (terminal 4) — single consumer for both subjects

```bash
python scripts/dashboard_subscriber.py
```

A matplotlib window opens with **two side-by-side panels**:
* **Left**: the most recent JPEG frame from `sensor.camera.rgb` (the RELLIS
  pylon camera).
* **Right**: the most recent BEVGrid from `perception.traversability.grid`
  (auto-zoomed, scalebar, risk colormap).

Each panel's title strip updates independently at its publisher's rate:

```
frame_id=frame_002800   pose_sigma=2.063m   received=14   latency≈3.2ms
```

The map redraws every 100 ms (configurable via `--refresh-ms`). pose_sigma climbs along the trajectory exactly as Ablation E showed offline — a live demonstration that uncertainty propagation survived end-to-end through the wire.

### 4. (Optional) Verify just the wire, no GUI

```bash
python -m transport.nats_adapter
# expected: self-test OK
```

Pure pub/sub round-trip with a 12-byte payload. If this passes but the dashboard misbehaves, the issue is in matplotlib / TkAgg, not NATS.

### 5. (Optional) Save received frames as PNGs for blog assets

```bash
python scripts/dashboard_subscriber.py --save-frames /tmp/m3_dashboard_frames/
ls /tmp/m3_dashboard_frames/
# frame_000050.npy, frame_000100.npy, ...
```

Then assemble with `ffmpeg` for the hero MP4:

```bash
ffmpeg -framerate 5 -pattern_type glob -i '/tmp/m3_dashboard_frames/*.npy.png' \
       -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
       results/m3/dashboard_demo.mp4
```

(You'll need a one-line numpy→PNG step; the dashboard saves `.npy` for losslessness.)

## Sanity checks if something's wrong

- **Dashboard window opens but stays blank.**
  - Publisher running? `tail` its log; `PUB ...` lines should be appearing.
  - Same `--subject` on both sides?
  - Same `--nats-url`? Default is `nats://localhost:4222`; change one without the other and they're talking past each other.

- **Publisher prints `Connection refused`.**
  - Broker isn't up. `docker compose ps` should show `nats` service alive, or `pgrep nats-server` should return a PID.

- **`ImportError: No module named 'transport.proto.perception_pb2'`.**
  - Run `bash transport/generate_protos.sh` first.

- **`ImportError: No module named 'nats'`.**
  - `pip install nats-py protobuf numpy matplotlib`.

- **Latency in dashboard reads negative or huge.**
  - Publisher and subscriber clocks differ. On the same host this should be sub-millisecond; if you're seeing seconds, something serialized weirdly.

## Files

| Path | Purpose |
|---|---|
| `transport/proto/perception.proto` | Schema (BEVGrid, Header, etc.). |
| `transport/proto/perception_pb2.py` | Generated bindings (don't edit). |
| `transport/generate_protos.sh` | Re-generate `_pb2.py` after editing `.proto`. |
| `transport/nats_adapter.py` | Async pub/sub wrapper around `nats-py`. |
| `scripts/publish_grid_stream.py` | Replays accumulator snapshots → NATS. |
| `scripts/dashboard_subscriber.py` | Live BEV-map matplotlib dashboard. |
| `docker/docker-compose.yml` | `nats` service definition (jetstream enabled). |
| `src/accumulator_runner.cpp:272,472` | C++ `--publish-nats` flag (parsed, stub). |
