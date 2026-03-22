# Architecture Documentation

## Transport Layer

Core perception services are **transport-agnostic**. The architecture separates:

1. **Core modules** — own the algorithms, produce typed outputs
2. **Transport** — moves bytes (NATS/gRPC for production, ROS2 DDS for demo)
3. **Adapters** — translate between ecosystems

### NATS Subject Taxonomy

| Subject | Publisher | Subscriber(s) | Rate |
|---------|-----------|---------------|------|
| `sensors.lidar.points` | replay | lidar_geometry, fusion | 5-20 Hz |
| `sensors.camera.rgb` | replay | camera_infer, ui | 10-30 Hz |
| `perception.traversability.grid` | lidar_geometry/fusion | planner, ui, safety | 5-20 Hz |
| `perception.objects.detections` | camera_infer | tracking, fusion | 10-30 Hz |
| `perception.objects.tracks` | tracking | safety, ui, planner | 10-30 Hz |
| `safety.cmd_vel.safe` | safety | vehicle_control | 10-50 Hz |
| `telemetry.metrics` | all | telemetry | 1-2 Hz |
| `telemetry.events` | safety | telemetry/ui | burst |

### Decision Table: NATS vs gRPC vs Kafka

- **NATS Core**: On-vehicle data plane bus. Low-latency event distribution.
- **NATS JetStream**: Durable audit logs, replay, safety events.
- **gRPC**: Control plane (GetStatus, SetConfig, DumpState).
- **Kafka**: Optional cloud bridge for fleet analytics + ML training.
- **ROS2/DDS**: Demo adapter for RViz and Nav2.

## Service Boundaries

See `transport/proto/*.proto` for message schemas.
