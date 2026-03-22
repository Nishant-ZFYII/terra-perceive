"""
nats_adapter.py — NATS pub/sub adapter for production transport.

YOUR TASK:
  1. Connect to NATS server (nats-py async client)
  2. Publish perception outputs on subjects from PDF Section 3.2:
     - perception.traversability.grid
     - perception.objects.detections
     - perception.objects.tracks
     - safety.cmd_vel.safe
     - telemetry.metrics
     - telemetry.events
  3. Serialize via protobuf (transport/proto/*.proto)
  4. JetStream for durable streams (safety events, metrics)

PDF reference: Part 3 (Addendum, pages 1-5)
Decision table: Part 5 (pages 5-6) — NATS vs gRPC vs Kafka
"""
