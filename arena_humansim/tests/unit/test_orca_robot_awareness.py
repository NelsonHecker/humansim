from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT, AgentPool
from arena_humansim.local_planner.orca import ORCAPlanner
from arena_humansim.utils.types import Pose2D


def test_orca_blind_pedestrian_ignores_robot(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = pool_empty(capacity=8)
    planner = ORCAPlanner()

    # Agent 1: Human at (0, 0), goal at (5, 0), blind to robot
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_awareness[0] = 0.0
    pool.desired_vel[0] = 1.0

    # Agent 2: Robot at (1.0, 0.0) right in front
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.desired_vel[1] = 0.0

    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)})

    planner.compute_pool(pool, dt=0.1)

    # Blind pedestrian keeps preferred forward velocity directly toward goal without swerving
    assert pool.vel[0, 0] > 0.95
    assert abs(pool.vel[0, 1]) < 1e-4


def test_orca_aware_pedestrian_swerves_or_decelerates_for_robot(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    pool = pool_empty(capacity=8)
    planner = ORCAPlanner()

    # Agent 1: Human at (0, 0), goal at (5, 0), fully aware of robot
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_awareness[0] = 1.0
    pool.desired_vel[0] = 1.0

    # Agent 2: Robot at (1.0, 0.0) blocking path
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.desired_vel[1] = 0.0

    pool.set_goals({1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)})

    planner.compute_pool(pool, dt=0.1)

    # Aware pedestrian either swerves (lateral velocity != 0) or decelerates
    assert abs(pool.vel[0, 1]) > 1e-3 or pool.vel[0, 0] < 0.9
