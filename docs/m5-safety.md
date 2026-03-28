---
layout: post
title: "M5: Kinematic Safety Supervisor"
date: 2026-03-28
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Kinematic Safety Supervisor

*Part of the Terra Perceive series.*

After M4, the pipeline can tell you the fused risk of every terrain cell: geometry from the traversability grid, semantics from SegFormer. What it cannot tell you is whether the robot should slow down, brake hard, or emergency-stop — and how urgently. A risk map is a spatial answer. A safety supervisor needs a *temporal* answer: given the robot's current speed and terrain, how much time do we have before something goes wrong?

This milestone replaces the common "if distance < 3m, stop" heuristic with a physics-based model. The stopping distance depends on velocity *and* terrain friction. The intervention depends on time-to-collision, not raw distance. The result is a system that knows a bulldozer on mud needs 8 metres to stop at 2 m/s, not 3.

---

## Stopping Distance from First Principles

The core equation comes from the work-energy theorem. A vehicle at velocity $$v$$ has kinetic energy $$\frac{1}{2}mv^2$$. The only force decelerating it is friction: $$F = \mu m g$$, where $$\mu$$ is the friction coefficient and $$g = 9.81\;\text{m/s}^2$$. Friction does work over distance $$d$$:

$$W_{\text{friction}} = -\mu m g \cdot d$$

Setting work equal to kinetic energy change (final KE = 0):

$$-\mu m g \cdot d = -\frac{1}{2}mv^2$$

Mass cancels — a bicycle and a bulldozer at the same speed and friction have the same stopping *distance* (though very different stopping *forces*). Solving for $$d$$:

$$d_{\text{brake}} = \frac{v^2}{2 \mu g}$$

But braking doesn't start instantly. There is a sensor-to-actuator latency $$t_{\text{react}}$$ (we use 0.2 s) during which the vehicle continues at full speed:

$$d_{\text{react}} = v \cdot t_{\text{react}}$$

Total stopping distance:

$$\boxed{d_{\text{stop}} = \frac{v^2}{2 \mu g} + v \cdot t_{\text{react}}}$$

The braking term is quadratic in $$v$$ — doubling speed quadruples the braking distance. The reaction term is linear. At highway speeds the braking term dominates; at the low speeds of construction equipment (2–5 m/s), the reaction term is a significant fraction of the total.

**Hand-calculated checkpoints:**

| $$v$$ (m/s) | $$\mu$$ | $$d_{\text{brake}}$$ (m) | $$d_{\text{react}}$$ (m) | $$d_{\text{stop}}$$ (m) |
|:-----------:|:-------:|:------------------------:|:------------------------:|:-----------------------:|
| 2.0 | 0.6 | 0.34 | 0.40 | **0.74** |
| 5.0 | 0.6 | 2.12 | 1.00 | **3.12** |
| 2.0 | 0.3 | 0.68 | 0.40 | **1.08** |

The third row shows why terrain matters: halving $$\mu$$ (from good gravel to wet mud) nearly doubles the stopping distance at the same speed.

---

## Terrain-Aware Friction

The friction coefficient $$\mu$$ is not a constant — it depends on the terrain the robot is currently driving over. M3's traversability grid already computes a per-cell score in $$[0, 1]$$ from slope, roughness, and step height. We close the loop by mapping that score to friction:

$$\mu = \mu_{\text{base}} + \mu_{\text{scale}} \cdot s_{\text{trav}}$$

With $$\mu_{\text{base}} = 0.3$$ and $$\mu_{\text{scale}} = 0.5$$:

| Terrain | $$s_{\text{trav}}$$ | $$\mu$$ | Physical meaning |
|---------|:-------------------:|:-------:|-----------------|
| Wet mud / steep slope | 0.0 | 0.3 | Worst case — long stopping distance |
| Mixed terrain | 0.5 | 0.55 | Moderate grip |
| Dry gravel / flat | 1.0 | 0.8 | Best case — short stopping distance |

This means the safety supervisor automatically becomes more conservative on dangerous terrain. A cell with high slope and high roughness produces a low traversability score, which produces a low $$\mu$$, which produces a longer $$d_{\text{stop}}$$, which triggers interventions earlier. The chain is fully automatic — no terrain-specific tuning required.

---

## Time-to-Collision

Raw distance to an obstacle is a poor trigger for safety interventions. A worker 5 metres away is safe if the robot is crawling at 0.5 m/s, but dangerous if it's moving at 5 m/s. What matters is *time*: how many seconds until the gap closes to within the stopping distance?

$$\text{TTC} = \frac{d_{\text{worker}} - d_{\text{stop}}}{v_{\text{relative}}}$$

where $$v_{\text{relative}} = v_{\text{vehicle}} - v_{\text{worker}}$$ is the closing speed. The sign convention: a worker moving *away* from the robot (positive $$v_{\text{worker}}$$) reduces $$v_{\text{relative}}$$, increasing TTC. A worker moving *toward* the robot (negative $$v_{\text{worker}}$$) increases closing speed.

**Edge cases:**

| Condition | TTC | Meaning |
|-----------|-----|---------|
| $$v_{\text{relative}} \leq 0$$ | $$+\infty$$ | Objects diverging — gap is growing, safe |
| $$d_{\text{worker}} > d_{\text{stop}}$$, $$v_{\text{relative}} > 0$$ | Positive, finite | Closing — seconds until collision |
| $$d_{\text{worker}} < d_{\text{stop}}$$ | Negative | Already inside stopping distance — cannot stop in time |

The negative TTC case is the most critical: it means braking *right now* at maximum friction will not prevent a collision. This is physically distinct from "TTC is small" — it requires an immediate emergency stop, not a gradual slowdown.

---

## Priority-Ordered Interventions

Safety systems use a strict priority hierarchy, not a flat if-else (ISO 26262). Higher-severity interventions always override lower ones:

| Priority | Condition | Action | Scale factor |
|:--------:|-----------|--------|:------------:|
| **P0** | Sensor timeout > 200 ms | Emergency stop | 0.0 |
| **P0** | TTC $$\leq$$ 0 | Emergency stop | 0.0 |
| **P1** | TTC < 2.0 s | Hard brake | 0.1 |
| **P2** | TTC < 5.0 s | Proportional scale | $$\frac{\text{TTC} - 2.0}{5.0 - 2.0}$$ |
| — | TTC $$\geq$$ 5.0 s | No intervention | 1.0 |

The sensor timeout check comes first — before any TTC computation. If the LiDAR hasn't produced data in 200 ms, all distance measurements are stale. The TTC calculation would be based on outdated information, so the correct action is an immediate stop regardless of what the numbers say.

The proportional scaling band (P2) provides smooth deceleration between 2 s and 5 s. At TTC = 3.5 s, the scale factor is $$(3.5 - 2.0) / (5.0 - 2.0) = 0.5$$, halving the commanded velocity. At TTC = 2.1 s, it drops to ~3% — nearly a full stop. This avoids the discontinuity of jumping from "full speed" to "hard brake" at a single threshold.

---

## Event Logging

Every call to the safety supervisor produces a `SafetyEvent` record:

```
timestamp, rule, d_worker, d_stop, TTC, friction_mu, vel_before, vel_after
```

Example output:

```
0.00, TTC >= 5s,      50.00, 0.74, 24.63, 0.60, 2.00, 2.00
0.10, TTC < 5s,        6.26, 0.74, 2.76,  0.60, 2.00, 0.51
0.20, TTC < 2s,        1.50, 0.74, 0.38,  0.60, 2.00, 0.20
0.30, d_worker < d_stop, 0.50, 0.74, -0.12, 0.60, 2.00, 0.00
```

The CSV log serves two purposes: (1) debugging — you can trace exactly which rule fired and why at every timestep; (2) safety audit — in a production system, this log feeds into a NATS JetStream topic for post-incident replay (planned for Phase 2).

---

## Latency Self-Monitoring

The safety loop must run faster than the control loop it feeds. We target < 5 ms per iteration on mock data (at 50 Hz control rate, the safety supervisor gets 20 ms per cycle — 5 ms leaves ample margin for the rest of the pipeline).

Each `evaluate()` call is timed with `std::chrono::high_resolution_clock`. The durations are stored in a vector; at the end of a session, sorting the vector and indexing at the 50th and 95th percentiles gives p50 and p95 latency:

- **p50** = median latency (typical case)
- **p95** = 95th percentile (worst 5% of calls)

If p50 is 0.1 ms but p95 is 8 ms, the system has occasional spikes — possibly from memory allocation in the event log's `push_back`. A production version would pre-allocate the vector or use a ring buffer.

---

## Test Results

10 new unit tests covering all sub-goals, all passing:

| Test | What it verifies |
|------|-----------------|
| `StoppingDistanceKnownValues` | d_stop matches hand-calculated values within 1% |
| `TTCApproachingWorker` | Stationary worker at 10 m, TTC = 4.63 s |
| `TTCRecedingWorker` | Worker moving away, TTC = infinity |
| `TTCAlreadyTooClose` | Worker at 0.5 m, TTC < 0 |
| `InterventionEmergencyStop` | d_worker < d_stop triggers E-stop, scale = 0.0 |
| `InterventionHardBrake` | TTC < 2 s triggers hard brake, scale = 0.1 |
| `InterventionProportionalScale` | TTC = 3.5 s, scale = 0.5 |
| `InterventionNone` | Worker at 50 m, no intervention, scale = 1.0 |
| `FrictionFromTraversability` | trav=0 maps to mu=0.3, trav=1 maps to mu=0.8 |
| `LowFrictionIncreasesStoppingDistance` | mu=0.3 produces longer d_stop than mu=0.8 |

**Total test count: 52/52 (42 prior + 10 safety).**

---

## What I'd Do Differently

- **Forward-arc geometric filtering.** The current implementation takes `d_to_nearest_worker` as a scalar input — the caller is responsible for filtering obstacles to the forward wedge. A more complete version would take a list of obstacle positions and internally filter to a configurable ±30 arc, 15 m range. This would make the supervisor self-contained rather than dependent on the caller's geometry.
- **Multiple obstacle handling.** The current API evaluates one obstacle at a time. A production version would iterate over all obstacles in the forward arc and use the *minimum* TTC to drive the intervention — the most dangerous obstacle wins.
- **Pre-allocated event log.** Using `std::vector::push_back` in a safety-critical loop is technically unbounded in latency due to reallocation. A ring buffer with a fixed capacity would guarantee constant-time insertion.

---

## Connection to Future Steps

M5 closes the perception-to-safety loop that is the core of Phase 1:

```
LiDAR scan
  --> RANSAC ground segmentation (M2)
    --> Traversability grid: risk + confidence (M3)
      --> Camera-LiDAR fusion: semantic override (M4)
        --> Terrain friction: mu = f(traversability) (M5.4)
          --> Stopping distance: d_stop = f(v, mu) (M5.1)
            --> TTC + intervention (M5.2, M5.3)
              --> Safety event log (M5.5)
```

Every component feeds the next. The traversability score from M3 determines how much friction the safety supervisor assumes. A muddy slope (high risk, low traversability) produces a low $$\mu$$, which produces a long stopping distance, which triggers earlier interventions — exactly the conservative behaviour you want on dangerous terrain.

M6 wires this pipeline into a Docker container with a smoke test: `docker-compose up` processes a bundled RELLIS-3D sample and produces a BEV image, a safety CSV, and a timing report.

---

*Code: `src/safety_supervisor.cpp`, `include/safety_supervisor.hpp`, `tests/cpp/test_safety.cpp`*

---

## References

1. OpenStax, *"Kinetic Energy and the Work-Energy Theorem"*, College Physics 1e, Ch. 7.2 — derivation of stopping distance from the work-energy theorem.
2. Criticality Metrics Documentation, [*"Time To Collision (TTC)"*](https://criticality-metrics.readthedocs.io/en/latest/time-scale/TTC.html) — formal TTC definition, handling of diverging objects, comparison with other safety metrics.
3. Nav2 Documentation, [*"Collision Monitor"*](https://docs.nav2.org/configuration/packages/configuring-collision-monitor.html) — production ROS2 pattern for polygon-based detection zones and Stop/Slowdown/Approach models.
4. ISO 26262, *"Road vehicles — Functional safety"* — ASIL classification framework and intervention priority hierarchy for safety-critical automotive systems.
5. Matias Wermelinger et al., *"Navigation Planning for Legged Robots in Challenging Terrain"*, IROS 2016 — vehicle-aware traversability scoring with multiplicative penalty model.
6. Shrey Aeron et al., *"RoadRunner — Learning Traversability Estimation for Autonomous Off-road Driving"*, arXiv 2402.19341, 2024 — traversability score to friction coefficient mapping for off-road navigation.
