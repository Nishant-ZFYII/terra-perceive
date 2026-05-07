"""
safety_replay.py — Python port of the SafetySupervisor decision logic for
audit-trail replay (P2-M5.A5).

This is a faithful re-implementation of the kinematic rules in
`include/safety_supervisor.hpp` + `src/safety_supervisor.cpp`, in the same
units and with the same constants:

    d_stop(v, mu)   = v^2 / (2 mu g) + v * t_react
    mu(trav)        = mu_base + mu_scale * trav
    TTC             = (d_worker - d_stop) / v_relative

Intervention rules (priority order):
    TTC <= 0  or  d_worker < d_stop     -> "d_worker < d_stop"     (E-STOP)
    TTC <  ttc_hard_brake               -> "TTC < 2s"              (HARD BRAKE)
    TTC <  ttc_proportional             -> "TTC < 5s"              (proportional)
    else                                -> "TTC >= 5s"             (no action)

Why re-implement instead of binding to the C++ class:
  - Pybind11 binding would be its own milestone; out of scope for M5.A5.
  - The rule set is small enough (<50 LOC) to faithfully port.
  - The replay harness is fundamentally about *deciding whether the rule
    that fired in the past would still fire under today's logic* — having
    the rule logic in pure Python makes the replay harness self-contained.

If the C++ rules change, this file has to change too. There's a unit test
(test_replay_safety_events.py) that asserts ports stay synced via
hand-checked vectors at known (v, d, v_w, trav) inputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyConfig:
    gravity: float = 9.81
    reaction_time_s: float = 0.2
    mu_base: float = 0.3
    mu_scale: float = 0.5
    ttc_hard_brake: float = 2.0
    ttc_proportional: float = 5.0


def stopping_distance(v: float, mu: float, cfg: SafetyConfig = SafetyConfig()) -> float:
    return (v * v) / (2.0 * mu * cfg.gravity) + v * cfg.reaction_time_s


def trav_to_mu(trav: float, cfg: SafetyConfig = SafetyConfig()) -> float:
    return cfg.mu_base + cfg.mu_scale * max(0.0, min(trav, 1.0))


def evaluate(
    v_vehicle: float,
    d_worker: float,
    v_worker: float,
    trav_score: float,
    cfg: SafetyConfig = SafetyConfig(),
) -> tuple[str, float]:
    """Return (rule_string, recommended_vel_scale_factor).

    rule_string matches the SafetyEvent.rule field that the C++ supervisor
    publishes, so a replay can compare past vs present by string equality.
    """
    mu = trav_to_mu(trav_score, cfg)
    d_stop = stopping_distance(v_vehicle, mu, cfg)
    v_rel = v_vehicle - v_worker

    if d_worker < d_stop:
        return ("d_worker < d_stop", 0.0)
    if v_rel <= 0.0:
        return ("TTC >= 5s", 1.0)

    ttc = (d_worker - d_stop) / v_rel
    if ttc < 0.0:
        return ("d_worker < d_stop", 0.0)
    if ttc < cfg.ttc_hard_brake:
        return ("TTC < 2s", 0.1)
    if ttc < cfg.ttc_proportional:
        scale = (ttc - cfg.ttc_hard_brake) / (cfg.ttc_proportional - cfg.ttc_hard_brake)
        return ("TTC < 5s", scale)
    return ("TTC >= 5s", 1.0)
