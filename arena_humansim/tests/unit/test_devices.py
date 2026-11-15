"""Held assistive devices: the shipped agent types' devices posed over the gait cycle, and robots touching them."""

from __future__ import annotations

import math

import numpy as np
import pytest
from arena_humansim.core.agents import BUILTIN_AGENTS
from arena_humansim.core.agents.types import AgentType
from arena_humansim.core.devices import BODY, CANE, END, ROLLATOR, START, WALKING_FRAME, DeviceSpec, Disk, RobotContactMonitor, pose_device

PHASES = np.linspace(0.0, 2.0 * math.pi, 73)
HOLDER = 7


def _spec(name: str) -> DeviceSpec:
    spec = BUILTIN_AGENTS[name].device
    assert spec is not None
    return spec


def _holder(radius: float = 0.3) -> Disk:
    return Disk(HOLDER, 0.0, 0.0, 0.0, radius)


def _robot(x: float, y: float, radius: float = 0.2) -> Disk:
    return Disk(-1, x, y, math.pi, radius)


def test_shipped_types_hold_their_devices() -> None:
    held = {name: (t.device.kind, t.device.model) for name, t in BUILTIN_AGENTS.items() if t.device is not None}
    assert held == {
        "blind_white_cane": (CANE, "Common/Object/WhiteCane"),
        "blind_rolling_cane": (CANE, "Common/Object/RollingCane"),
        "elder_rollator": (ROLLATOR, "Common/Object/RollatorFitted"),
        "patient_walking_frame": (WALKING_FRAME, "Common/Object/WalkerFitted"),
    }


@pytest.mark.parametrize(("name", "lift"), [("blind_white_cane", 0.04), ("blind_rolling_cane", 0.0)])
def test_cane_tip_sweeps_the_path_ahead_of_a_fixed_fist(name: str, lift: float) -> None:
    poses = [pose_device(HOLDER, _spec(name), 0.0, 0.0, 0.0, w) for w in PHASES]
    grips, tips = np.array([p.start for p in poses]), np.array([p.end for p in poses])
    assert np.allclose(grips, [0.3, -0.1, 0.9], atol=0.005)
    assert 0.9 < tips[:, 0].min() and tips[:, 0].max() < 1.12
    assert tips[:, 1].min() < -0.29 and tips[:, 1].max() > 0.29  # two-point touch / constant contact across the path
    assert -0.01 < tips[:, 2].min() and tips[:, 2].max() < lift + 0.01
    assert np.allclose(np.linalg.norm(tips - grips, axis=1), 1.17, atol=0.015)  # the shaft below the fist keeps its length


def test_devices_turn_and_move_with_their_holder() -> None:
    spec = _spec("blind_white_cane")
    here, there = pose_device(HOLDER, spec, 0.0, 0.0, 0.0, 1.0), pose_device(HOLDER, spec, 2.0, -1.0, math.pi / 2, 1.0)
    assert np.allclose(there.end[:2], [2.0 - here.end[1], -1.0 + here.end[0]])
    assert there.end[2] == pytest.approx(here.end[2])


def test_rollator_and_walking_frame_stand_ahead_of_their_holder() -> None:
    rollator = pose_device(HOLDER, _spec("elder_rollator"), 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(rollator.centre[:2], [0.36, 0.0], atol=0.01)
    assert rollator.half == pytest.approx((0.344, 0.285), abs=0.001) and rollator.height == pytest.approx(0.88, abs=0.01)
    frames = [pose_device(HOLDER, _spec("patient_walking_frame"), 0.0, 0.0, 0.0, w) for w in PHASES]
    xs, lifts = [f.centre[0] for f in frames], [f.start[2] for f in frames]
    assert min(xs) == pytest.approx(0.25, abs=0.02) and max(xs) == pytest.approx(0.47, abs=0.02)  # carried forward, then stepped into
    assert max(lifts) > 0.07 and min(lifts) < 0.01


def test_robot_at_the_cane_tip_touches_the_cane_and_not_the_body() -> None:
    dev = pose_device(HOLDER, _spec("blind_rolling_cane"), 0.0, 0.0, 0.0, 1.0)
    ahead = (dev.end[:2] - dev.start[:2]) / np.linalg.norm(dev.end[:2] - dev.start[:2])
    x, y = dev.end[:2] + ahead * 0.18  # 2 cm overlap with a 0.2 m robot, the shaft 1.25 cm thick
    edges = RobotContactMonitor().step(0.0, [_robot(x, y)], [_holder()], {HOLDER: dev})
    assert [(e.event, e.part) for e in edges] == [(START, CANE)]
    assert edges[0].depth == pytest.approx(0.0325, abs=0.002)
    assert np.allclose(edges[0].point, dev.end, atol=1e-9)


def test_a_contact_ends_with_its_duration_and_deepest_overlap() -> None:
    dev = pose_device(HOLDER, _spec("elder_rollator"), 0.0, 0.0, 0.0, 0.0)
    front = 0.36 + 0.344
    monitor = RobotContactMonitor()
    assert [e.event for e in monitor.step(0.0, [_robot(front + 0.18, 0.0)], [_holder()], {HOLDER: dev})] == [START]
    assert monitor.step(0.5, [_robot(front + 0.15, 0.0)], [_holder()], {HOLDER: dev}) == []
    edges = monitor.step(1.25, [_robot(front + 0.5, 0.0)], [_holder()], {HOLDER: dev})
    assert [(e.event, e.part) for e in edges] == [(END, ROLLATOR)]
    assert edges[0].duration == pytest.approx(1.25) and edges[0].depth == pytest.approx(0.051, abs=0.002)
    assert not monitor.active


def test_rollator_and_body_contacts_are_told_apart() -> None:
    dev = pose_device(HOLDER, _spec("elder_rollator"), 0.0, 0.0, 0.0, 0.0)
    behind = RobotContactMonitor().step(0.0, [_robot(-0.45, 0.0)], [_holder()], {HOLDER: dev})
    assert [e.part for e in behind] == [BODY]
    inside = RobotContactMonitor().step(0.0, [_robot(0.55, 0.0)], [_holder()], {HOLDER: dev})  # between the rollator's legs
    assert [e.part for e in inside] == [ROLLATOR] and inside[0].depth == pytest.approx(0.2 + 0.155, abs=0.002)


def test_a_short_robot_passes_under_the_cane_shaft() -> None:
    dev = pose_device(HOLDER, _spec("blind_rolling_cane"), 0.0, 0.0, 0.0, 1.0)
    mid = (dev.start + dev.end) / 2  # 0.45 m above the floor
    robot = _robot(mid[0], mid[1], 0.08)
    assert {e.part for e in RobotContactMonitor().step(0.0, [robot], [_holder()], {HOLDER: dev})} == {CANE}
    assert {e.part for e in RobotContactMonitor(robot_height=0.3).step(0.0, [robot], [_holder()], {HOLDER: dev})} == set()
    assert {e.part for e in RobotContactMonitor(robot_height=0.6).step(0.0, [robot], [_holder()], {HOLDER: dev})} == {CANE}


def test_robot_radius_override_and_far_robots() -> None:
    dev = pose_device(HOLDER, _spec("elder_rollator"), 0.0, 0.0, 0.0, 0.0)
    far = _robot(0.36 + 0.344 + 0.4, 0.0)
    assert RobotContactMonitor().step(0.0, [far], [_holder()], {HOLDER: dev}) == []
    assert [e.part for e in RobotContactMonitor(robot_radius=0.45).step(0.0, [far], [_holder()], {HOLDER: dev})] == [ROLLATOR]
    assert RobotContactMonitor().step(0.0, [_robot(5.0, 0.0)], [_holder()], {HOLDER: dev}) == []


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "crutch"},
        {"kind": "cane", "grip": {"x": 0.3, "y": -0.1, "z": 0.9}, "tip": {"x": 1.0, "y": 0.0, "z": 0.0}},
        {"kind": "cane", "grip": {"x": 0.3, "y": -0.1}, "tip": {"x": 1.0, "y": 0.0, "z": 0.0}, "radius": 0.01},
        {"kind": "rollator", "origin": {"x": 0.36, "y": 0.0, "z": 0.0}, "size": {"length": 0.7, "width": 0.0, "height": 0.9}},
        {"kind": "walking_frame", "origin": {"x": 0.3, "y": 0.0, "z": 0.0}, "size": {"length": 0.4, "width": 0.5, "height": 0.7}, "colour": "grey"},
        {"kind": "rollator", "origin": {"x": {"amp": 0.1}, "y": 0.0, "z": 0.0}, "size": {"length": 0.7, "width": 0.6, "height": 0.9}},
    ],
)
def test_malformed_device_blocks_are_refused(raw: dict) -> None:
    with pytest.raises(ValueError):
        AgentType(name="aid_user", device=raw)
