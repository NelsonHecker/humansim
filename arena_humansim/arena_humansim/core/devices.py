"""Assistive devices agents hold (a long cane, a rollator, a walking frame) and robots touching them.

An agent type's `device:` block gives its device in the agent's body frame (x forward, y left, z up from the soles,
origin under the agent's position, metres) as tracks of the warped gait phase w, the convention of the gait profiles:
value = mean + sum_k amp_k * sin(k * w + phase_k), a plain number for a constant. The renderer poses the arm from the same
profile and phase, so the device contact is tested against is the one drawn in the hand.

  cane           grip, tip: the shaft from the fist's grip centre to the tip; radius: the shaft's
  rollator       origin: on the floor under the middle of the grips (z: lift); size: the footprint's length along the
  walking_frame  heading, width and height, and its centre `ahead` and `left` of the origin

and optionally `model`, the object model a renderer draws for the device (an Arena asset name).

The tracks are fitted by Arena (`python3 -m arena_simulation_setup.adapters.arena_animation_retarget.devices`), which poses
each type's gait profile on the character it was solved for.

RobotContactMonitor runs after collision resolution: every robot disk against every human's body disk and device, one
START edge when a contact begins and one END edge when it ends. Contact is tested in plan view; a robot height cuts a cane
to the part of its shaft low enough to touch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import attrs
import numpy as np

from arena_humansim.core.pool import AgentPool, PoolAware

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent

CANE = "cane"
ROLLATOR = "rollator"
WALKING_FRAME = "walking_frame"
KINDS = (CANE, ROLLATOR, WALKING_FRAME)
BOXES = (ROLLATOR, WALKING_FRAME)
BODY = "body"

START = 0
END = 1


@attrs.frozen
class Track:
    """mean + sum_k amp_k * sin(k * w + phase_k) of the warped gait phase w."""

    mean: float
    harmonics: tuple[tuple[float, float], ...] = ()

    @classmethod
    def parse(cls, raw: object, where: str) -> Track:
        if isinstance(raw, (int, float)):
            return cls(float(raw))
        if not isinstance(raw, Mapping) or set(raw) - {"mean", "harmonics"} or "mean" not in raw:
            raise ValueError(f"{where}: a number or {{mean, harmonics: [[amp, phase], ...]}}, got {raw!r}")
        harmonics = tuple((float(amp), float(ph)) for amp, ph in raw.get("harmonics") or ())
        return cls(float(raw["mean"]), harmonics)

    def at(self, w: float) -> float:
        return self.mean + sum(amp * math.sin(k * w + ph) for k, (amp, ph) in enumerate(self.harmonics, start=1))


@attrs.frozen
class Point3:
    """A body-frame point whose coordinates follow the gait phase."""

    x: Track
    y: Track
    z: Track

    @classmethod
    def parse(cls, raw: object, where: str) -> Point3:
        if not isinstance(raw, Mapping) or set(raw) != {"x", "y", "z"}:
            raise ValueError(f"{where}: needs exactly x, y and z, got {raw!r}")
        return cls(*(Track.parse(raw[a], f"{where}.{a}") for a in "xyz"))

    def at(self, w: float) -> np.ndarray:
        return np.array([self.x.at(w), self.y.at(w), self.z.at(w)])


_CANE_KEYS = {"kind", "grip", "tip", "radius"}
_BOX_KEYS = {"kind", "origin", "size"}
_OPTIONAL_KEYS = {"model"}
_SIZE_KEYS = {"length", "width", "height", "ahead", "left"}


@attrs.frozen
class DeviceSpec:
    """The device an agent type holds, in the body frame (module docstring)."""

    kind: str
    grip: Point3 | None = None
    tip: Point3 | None = None
    radius: float = 0.0
    origin: Point3 | None = None
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    ahead: float = 0.0
    left: float = 0.0
    model: str = ""

    @classmethod
    def parse(cls, raw: object) -> DeviceSpec:
        if isinstance(raw, DeviceSpec):
            return raw
        if not isinstance(raw, Mapping) or raw.get("kind") not in KINDS:
            raise ValueError(f"device: kind must be one of {', '.join(KINDS)}, got {raw!r}")
        kind = str(raw["kind"])
        model = str(raw.get("model", ""))
        keys = set(raw) - _OPTIONAL_KEYS
        if kind == CANE:
            if keys != _CANE_KEYS:
                raise ValueError(f"device ({kind}): needs exactly {', '.join(sorted(_CANE_KEYS))}[, model], got {', '.join(sorted(raw))}")
            radius = float(raw["radius"])
            if radius < 0.0:
                raise ValueError(f"device ({kind}): radius must not be negative, got {radius}")
            return cls(kind, grip=Point3.parse(raw["grip"], "device.grip"), tip=Point3.parse(raw["tip"], "device.tip"), radius=radius, model=model)
        size = raw.get("size")
        if keys != _BOX_KEYS or not isinstance(size, Mapping) or not {"length", "width", "height"} <= set(size) or set(size) - _SIZE_KEYS:
            raise ValueError(f"device ({kind}): needs exactly kind, origin and size {{length, width, height[, ahead, left]}}[, model], got {raw!r}")
        dims = {k: float(size.get(k, 0.0)) for k in _SIZE_KEYS}
        if min(dims["length"], dims["width"], dims["height"]) <= 0.0:
            raise ValueError(f"device ({kind}): length, width and height must be positive, got {dict(size)}")
        return cls(kind, origin=Point3.parse(raw["origin"], "device.origin"), model=model, **dims)


@attrs.frozen(eq=False)
class DevicePose:
    """A device posed in the world (map frame): a cane's shaft from `start` (fist) to `end` (tip), or a box with its origin at
    `start` (= `end`) and its footprint centred at `centre`, `half` its half length along `yaw` and half width."""

    agent_id: int
    kind: str
    start: np.ndarray
    end: np.ndarray
    centre: np.ndarray
    yaw: float
    radius: float = 0.0
    half: tuple[float, float] = (0.0, 0.0)
    height: float = 0.0
    model: str = ""


def pose_device(agent_id: int, spec: DeviceSpec, x: float, y: float, theta: float, w: float) -> DevicePose:
    """`spec` posed for an agent at (x, y) heading `theta` at warped gait phase `w`."""
    c, s = math.cos(theta), math.sin(theta)

    def world(p: np.ndarray) -> np.ndarray:
        return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])

    if spec.kind == CANE:
        assert spec.grip is not None and spec.tip is not None
        start, end = world(spec.grip.at(w)), world(spec.tip.at(w))
        return DevicePose(agent_id, spec.kind, start, end, end.copy(), math.atan2(end[1] - start[1], end[0] - start[0]), radius=spec.radius, model=spec.model)
    assert spec.origin is not None
    o = spec.origin.at(w)
    start = world(o)
    centre = world(np.array([o[0] + spec.ahead, o[1] + spec.left, 0.0]))
    return DevicePose(agent_id, spec.kind, start, start.copy(), centre, theta, half=(spec.length / 2, spec.width / 2), height=spec.height, model=spec.model)


def cane_contact(cx: float, cy: float, r: float, dev: DevicePose, max_z: float = math.inf) -> tuple[float, np.ndarray] | None:
    """(depth, point) where a disk at (cx, cy) of radius `r` overlaps the plan view of a cane's shaft cut at height `max_z`,
    None without overlap. The point is the shaft's nearest to the disk centre."""
    p0, p1 = dev.start, dev.end
    if p0[2] > max_z and p1[2] > max_z:
        return None
    if p0[2] > max_z or p1[2] > max_z:
        s = (max_z - p0[2]) / (p1[2] - p0[2])
        cut = p0 + s * (p1 - p0)
        p0, p1 = (cut, p1) if p0[2] > max_z else (p0, cut)
    d = p1[:2] - p0[:2]
    n2 = float(d @ d)
    t = 0.0 if n2 == 0.0 else min(1.0, max(0.0, float((np.array([cx, cy]) - p0[:2]) @ d) / n2))
    q = p0 + t * (p1 - p0)
    depth = r + dev.radius - math.hypot(cx - q[0], cy - q[1])
    return (depth, q) if depth >= 0.0 else None


def box_contact(cx: float, cy: float, r: float, dev: DevicePose) -> tuple[float, np.ndarray] | None:
    """(depth, point) where a disk at (cx, cy) of radius `r` overlaps a box's footprint, None without overlap. The point is
    the footprint's nearest to the disk centre, the centre itself once inside."""
    c, s = math.cos(dev.yaw), math.sin(dev.yaw)
    dx, dy = cx - dev.centre[0], cy - dev.centre[1]
    u, v = c * dx + s * dy, -s * dx + c * dy
    hl, hw = dev.half
    qu, qv = min(hl, max(-hl, u)), min(hw, max(-hw, v))
    gap = math.hypot(u - qu, v - qv)
    depth = r + min(hl - abs(u), hw - abs(v)) if gap == 0.0 else r - gap
    point = np.array([dev.centre[0] + c * qu - s * qv, dev.centre[1] + s * qu + c * qv, 0.0])
    return (depth, point) if depth >= 0.0 else None


def body_contact(cx: float, cy: float, r: float, x: float, y: float, radius: float) -> tuple[float, np.ndarray] | None:
    """(depth, point) where a disk at (cx, cy) of radius `r` overlaps a body disk at (x, y), None without overlap. The point
    is on the body's rim toward the robot."""
    d = math.hypot(cx - x, cy - y)
    depth = r + radius - d
    if depth < 0.0:
        return None
    ux, uy = ((cx - x) / d, (cy - y) / d) if d > 0.0 else (1.0, 0.0)
    return depth, np.array([x + radius * ux, y + radius * uy, 0.0])


@attrs.frozen
class Disk:
    """A pool row in plan view: id, position, heading and radius."""

    agent_id: int
    x: float
    y: float
    theta: float
    radius: float


@attrs.frozen(eq=False)
class ContactEdge:
    """One edge (START or END) of a robot touching an agent's part."""

    event: int
    part: str
    robot: Disk
    agent: Disk
    point: np.ndarray
    depth: float
    duration: float
    device: DevicePose | None


@attrs.define
class _Contact:
    start_t: float
    robot: Disk
    agent: Disk
    point: np.ndarray
    depth: float
    deepest: float
    device: DevicePose | None


class RobotContactMonitor:
    """Robot-agent contacts per part (body or held device), reported as START and END edges.

    `robot_radius` > 0 replaces the robots' own disk radius, `robot_height` > 0 cuts canes above it."""

    def __init__(self, robot_radius: float = 0.0, robot_height: float = 0.0) -> None:
        self.robot_radius = robot_radius
        self.robot_height = robot_height
        self._active: dict[tuple[int, int, str], _Contact] = {}

    @property
    def active(self) -> Mapping[tuple[int, int, str], _Contact]:
        """Ongoing contacts by (robot id, agent id, part)."""
        return self._active

    def reset(self) -> None:
        self._active.clear()

    def touching(self, robots: Iterable[Disk], humans: Iterable[Disk], devices: Mapping[int, DevicePose]) -> dict[tuple[int, int, str], tuple[Disk, Disk, float, np.ndarray, DevicePose | None]]:
        """Every robot-part overlap now: {(robot id, agent id, part): (robot, agent, depth, point, device)}."""
        max_z = self.robot_height if self.robot_height > 0.0 else math.inf
        humans = list(humans)
        out: dict[tuple[int, int, str], tuple[Disk, Disk, float, np.ndarray, DevicePose | None]] = {}
        for robot in robots:
            r = self.robot_radius if self.robot_radius > 0.0 else robot.radius
            for agent in humans:
                dev = devices.get(agent.agent_id)
                if math.hypot(robot.x - agent.x, robot.y - agent.y) > r + agent.radius + _reach(dev, agent):
                    continue
                hit = body_contact(robot.x, robot.y, r, agent.x, agent.y, agent.radius)
                if hit is not None:
                    out[(robot.agent_id, agent.agent_id, BODY)] = (robot, agent, hit[0], hit[1], dev)
                if dev is None:
                    continue
                hit = cane_contact(robot.x, robot.y, r, dev, max_z) if dev.kind == CANE else box_contact(robot.x, robot.y, r, dev)
                if hit is not None:
                    out[(robot.agent_id, agent.agent_id, dev.kind)] = (robot, agent, hit[0], hit[1], dev)
        return out

    def step(self, t: float, robots: Iterable[Disk], humans: Iterable[Disk], devices: Mapping[int, DevicePose]) -> list[ContactEdge]:
        """Edges of this tick at sim time `t`: START for each new contact, END (its last touch) for each that ended."""
        now = self.touching(robots, humans, devices)
        edges: list[ContactEdge] = []
        for key in [k for k in self._active if k not in now]:
            c = self._active.pop(key)
            edges.append(ContactEdge(END, key[2], c.robot, c.agent, c.point, c.deepest, t - c.start_t, c.device))
        for key, (robot, agent, depth, point, dev) in now.items():
            c = self._active.get(key)
            if c is None:
                self._active[key] = _Contact(t, robot, agent, point, depth, depth, dev)
                edges.append(ContactEdge(START, key[2], robot, agent, point, depth, 0.0, dev))
            else:
                c.robot, c.agent, c.point, c.depth, c.device = robot, agent, point, depth, dev
                c.deepest = max(c.deepest, depth)
        return edges


def _reach(dev: DevicePose | None, agent: Disk) -> float:
    """How far beyond the agent's body disk its device can reach from its position, a cheap bound to skip far pairs."""
    if dev is None:
        return 0.0
    if dev.kind == CANE:
        return max(math.hypot(p[0] - agent.x, p[1] - agent.y) for p in (dev.start, dev.end)) + dev.radius
    return math.hypot(dev.centre[0] - agent.x, dev.centre[1] - agent.y) + math.hypot(*dev.half)


class DeviceExtension(PoolAware):
    """Per-row device specs of the pool's agents, and their devices posed from the gait phase."""

    _pool: AgentPool

    def attach(self, pool: AgentPool) -> None:
        self._pool = pool
        self.specs: list[DeviceSpec | None] = [None] * pool.capacity
        pool.register_extension(self)

    def on_pool_grow(self, new_capacity: int, old_capacity: int) -> None:
        self.specs.extend([None] * (new_capacity - len(self.specs)))

    def on_pool_add(self, idx: int, agent: BaseAgent) -> None:
        self.specs[idx] = agent.params.device

    def on_pool_update(self, idx: int, agent: BaseAgent) -> None:
        self.specs[idx] = agent.params.device

    def on_pool_swap(self, idx: int, last: int) -> None:
        self.specs[idx] = self.specs[last]
        self.specs[last] = None

    def on_pool_reset(self) -> None:
        self.specs = [None] * len(self.specs)

    def rows(self) -> np.ndarray:
        return np.array([i for i in range(self._pool.n) if self.specs[i] is not None], dtype=np.int64)

    def pose(self, rows: np.ndarray, warped_phase: Sequence[float]) -> dict[int, DevicePose]:
        """{agent id: posed device} of `rows`, `warped_phase` per row."""
        pool = self._pool
        out: dict[int, DevicePose] = {}
        for row, w in zip(rows, warped_phase, strict=True):
            spec = self.specs[int(row)]
            assert spec is not None
            aid = int(pool.agent_ids[row])
            out[aid] = pose_device(aid, spec, float(pool.pos[row, 0]), float(pool.pos[row, 1]), float(pool.theta[row]), float(w))
        return out


def as_device(raw: object) -> DeviceSpec | None:
    """attrs converter of an agent type's `device:` block."""
    return None if raw is None else DeviceSpec.parse(raw)
