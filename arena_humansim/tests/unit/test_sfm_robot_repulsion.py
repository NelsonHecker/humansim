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
    robot_x: float = 0.5,
) -> tuple[AgentPool, SFMPlanner]:
    pool = pool_empty(capacity=8)
    planner = SFMPlanner()
    planner.attach(pool)

    # Agent 1 (human) at (0, 0), moving towards (10, 0)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_repulsion_gain[0] = robot_repulsion_gain
    pool.policy_idx[0] = 0

    # Agent 2 (robot) at (robot_x, 0.0) blocking path
    pool.add_agent(agent_factory(agent_id=2, x=robot_x, y=0.0))
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


def test_sfm_reaction_gain_modulates_interaction_range(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    # Test that robot_reaction_gain scales the interaction range B
    # Place robot at 0.8m (beyond touching distance 0.5m) so (r_ij - d) < 0
    pool_early, planner_early = _setup_sfm_pool(pool_empty, agent_factory, robot_repulsion_gain=1.0, robot_x=0.8)
    pool_early.robot_reaction_gain[0] = 1.5  # early anticipation / longer interaction range

    pool_delayed, planner_delayed = _setup_sfm_pool(pool_empty, agent_factory, robot_repulsion_gain=1.0, robot_x=0.8)
    pool_delayed.robot_reaction_gain[0] = 0.5  # sluggish / short interaction range

    _, f_rep_early, _, _ = planner_early._compute_forces_pool(pool_early)
    _, f_rep_delayed, _, _ = planner_delayed._compute_forces_pool(pool_delayed)

    # At distance 0.5m, the early-reacting pedestrian feels more repulsive force than the delayed one
    assert abs(f_rep_early[0, 0]) > abs(f_rep_delayed[0, 0])
