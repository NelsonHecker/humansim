from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT, AgentPool
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.utils.types import Pose2D


def _setup_sfm_pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    robot_repulsion_gain: float,
) -> tuple[AgentPool, SFMPlanner]:
    pool = pool_empty(capacity=8)
    planner = SFMPlanner()
    planner.attach(pool)

    # Agent 1 (human) at (0, 0), moving towards (10, 0)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_repulsion_gain[0] = robot_repulsion_gain
    pool.policy_idx[0] = 0

    # Agent 2 (robot) at (0.5, 0.0) blocking path
    pool.add_agent(agent_factory(agent_id=2, x=0.5, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.policy_idx[1] = 0

    pool.set_goals({1: Pose2D(x=10.0, y=0.0)})
    indptr = np.array([0, 1, 1], dtype=np.int32)
    indices = np.array([1], dtype=np.int32)
    pool.set_neighbor_csr(indptr, indices)
    return pool, planner


def test_sfm_scared_repulsion_stronger_than_normal(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    normal_pool, normal_planner = _setup_sfm_pool(pool_empty, agent_factory, robot_repulsion_gain=1.0)
    normal_planner.compute_pool(normal_pool, store_forces=False, dt=0.05)

    scared_pool, scared_planner = _setup_sfm_pool(pool_empty, agent_factory, robot_repulsion_gain=2.0)
    scared_planner.compute_pool(scared_pool, store_forces=False, dt=0.05)

    # Scared pedestrian experiences stronger backward repulsion force, yielding slower forward velocity
    assert scared_pool.vel[0, 0] < normal_pool.vel[0, 0]


def test_sfm_aggressive_repulsion_weaker_than_normal(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    normal_pool, normal_planner = _setup_sfm_pool(pool_empty, agent_factory, robot_repulsion_gain=1.0)
    normal_planner.compute_pool(normal_pool, store_forces=False, dt=0.05)

    aggressive_pool, aggressive_planner = _setup_sfm_pool(pool_empty, agent_factory, robot_repulsion_gain=0.4)
    aggressive_planner.compute_pool(aggressive_pool, store_forces=False, dt=0.05)

    # Aggressive pedestrian yields less, maintaining higher forward velocity
    assert aggressive_pool.vel[0, 0] > normal_pool.vel[0, 0]


def test_sfm_zero_gain_repulsion_ignores_robot(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    zero_pool, zero_planner = _setup_sfm_pool(pool_empty, agent_factory, robot_repulsion_gain=0.0)
    f_att, f_rep, f_wall, _ = zero_planner._compute_forces_pool(zero_pool)

    # Repulsion force on agent 0 from robot should be exactly 0
    assert np.allclose(f_rep[0], 0.0)
