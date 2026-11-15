"""Robot contacts: a robot driven through world_state into a blind cane user's path is reported touching the cane first,
then the body, and each contact ends when the robot backs off."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentDevices as AgentDevicesMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import RobotContact as RobotContactMsg
from arena_humansim_msgs.msg import RobotContacts as RobotContactsMsg
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock

from ._capture import STREAM, settle, subscribe

ROBOT, PERSON = 1, 2
APPROACH, RETREAT = 100, 40


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="robot_contacts",
        simulation=SimulationParams(seed=5, dt=0.05, bt_tick_interval=5, max_ticks=APPROACH + RETREAT),
        modules=ModuleConfig(global_planner="navmesh"),
        agents=[
            AgentConfig(agent_id=ROBOT, kind=1, agent_type="robot", spawn_pose=Pose2D(x=3.5, y=0.0, theta=math.pi)),
            AgentConfig(agent_id=PERSON, agent_type="blind_white_cane", spawn_pose=Pose2D(x=0.0, y=0.0), goal_sequence=[Pose2D(x=12.0, y=0.0)], desired_velocity=0.75, robot_aware=False),
        ],
    )


def _stamp(tick: int, dt: float) -> Time:
    ns = round(tick * dt * 1e9)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _robot_at(tick: int, dt: float, x: float) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    msg.header.stamp = _stamp(tick, dt)
    robot = AgentStateMsg()
    robot.agent_id = 515151
    robot.pose = Pose2DMsg(x=x, y=0.0, theta=math.pi)
    robot.radius = 0.2
    msg.agents.append(robot)
    return msg


def test_robot_touches_the_cane_before_the_body_and_both_contacts_end(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario(), node_name="test_robot_contacts")
    contacts = subscribe(mgr, RobotContactsMsg, "robot_contacts", STREAM)
    devices = subscribe(mgr, AgentDevicesMsg, "agent_devices", STREAM)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)

    x = 3.5
    for tick in range(APPROACH + RETREAT):
        x += -0.3 * mgr._dt if tick < APPROACH else 1.0 * mgr._dt  # toward the person, then back off
        mgr._world_state_callback(_robot_at(tick, mgr._dt, x))
        mgr._subsystem_timer_callback(Clock(clock=_stamp(tick + 1, mgr._dt)))  # one tick, stamped with sim time
        executor.spin_once(timeout_sec=0.0)
    settle(executor)

    edges: list[RobotContactMsg] = [c for frame in contacts for c in frame.contacts]
    starts = [c for c in edges if c.event == RobotContactMsg.START]
    ends = [c for c in edges if c.event == RobotContactMsg.END]
    parts = [c.part for c in starts]
    assert parts[0] == "cane" and "body" in parts  # the cane reaches the robot first
    assert {c.part for c in ends} == {"cane", "body"}
    assert all(c.agent_id == PERSON and c.robot_id == ROBOT and c.agent_type == "blind_white_cane" for c in edges)

    cane = starts[0]
    p, a = cane.point, cane.agent_pose
    ahead = math.cos(a.theta) * (p.x - a.x) + math.sin(a.theta) * (p.y - a.y)
    assert 0.85 < ahead < 1.15 and p.z < 0.3  # the shaft's low end near the tip, sweeping across the robot's path
    assert 0.0 <= cane.depth < 0.1 and cane.device.kind == "cane" and cane.duration == 0.0
    assert all(c.duration > 0.0 and c.depth >= 0.0 for c in ends)

    assert devices and [d.kind for d in devices[-1].devices] == ["cane"] and devices[-1].devices[0].agent_id == PERSON


def test_robot_contact_params_apply_at_once_and_refuse_negatives(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario(), node_name="test_robot_contact_params")
    assert all(r.successful for r in mgr.set_parameters([Parameter("robot_contact.robot_radius", value=0.17), Parameter("robot_contact.robot_height", value=0.35)]))
    assert (mgr._robot_contacts.robot_radius, mgr._robot_contacts.robot_height) == (0.17, 0.35)
    assert not mgr.set_parameters([Parameter("robot_contact.robot_radius", value=-0.1)])[0].successful
    assert mgr._robot_contacts.robot_radius == 0.17
