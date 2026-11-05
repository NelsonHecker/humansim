from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT, AgentPool
from arena_humansim.perception.default import DefaultPerception


def test_blind_human_does_not_perceive_robot(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    perception = DefaultPerception()

    # Agent 1: Human at (0, 0), completely blind to robots
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_awareness[0] = 0.0

    # Agent 2: Robot at (1.0, 0.0)
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT

    # Agent 3: Another Human at (0.0, 1.0)
    pool.add_agent(agent_factory(agent_id=3, x=0.0, y=1.0))
    pool.kind[2] = KIND_HUMAN

    perception.compute_pool(pool)

    # Agent 1 (human) should see Agent 3 (human), but NOT Agent 2 (robot)
    start = int(pool.neighbor_indptr[0])
    stop = int(pool.neighbor_indptr[1])
    neighbors_of_1 = list(pool.neighbor_indices[start:stop])

    assert 1 not in neighbors_of_1, "Blind human should not perceive robot at index 1"
    assert 2 in neighbors_of_1, "Blind human should still perceive human at index 2"


def test_aware_human_perceives_both_robot_and_human(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    perception = DefaultPerception()

    # Agent 1: Normal human at (0, 0), aware of robots
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_awareness[0] = 1.0

    # Agent 2: Robot at (1.0, 0.0)
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT

    # Agent 3: Human at (0.0, 1.0)
    pool.add_agent(agent_factory(agent_id=3, x=0.0, y=1.0))
    pool.kind[2] = KIND_HUMAN

    perception.compute_pool(pool)

    start = int(pool.neighbor_indptr[0])
    stop = int(pool.neighbor_indptr[1])
    neighbors_of_1 = list(pool.neighbor_indices[start:stop])

    assert 1 in neighbors_of_1, "Normal human should perceive robot"
    assert 2 in neighbors_of_1, "Normal human should perceive human"


def test_distracted_human_has_constricted_vision_range(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    perception = DefaultPerception()

    # Agent 1: Distracted human at (0, 0), vision_range=5.0, awareness=0.3 -> effective range=1.5m
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.vision_range[0] = 5.0
    pool.proximity_sense[0] = 0.5
    pool.robot_awareness[0] = 0.3

    # Robot at 2.5m (outside 1.5m, but within 5.0m)
    pool.add_agent(agent_factory(agent_id=2, x=2.5, y=0.0))
    pool.kind[1] = KIND_ROBOT

    # Human at (2.0, 1.0) -> dist ~2.23m, angle ~26.5 deg (well inside 180 FOV and 5.0m range)
    pool.add_agent(agent_factory(agent_id=3, x=2.0, y=1.0))
    pool.kind[2] = KIND_HUMAN

    perception.compute_pool(pool)

    start = int(pool.neighbor_indptr[0])
    stop = int(pool.neighbor_indptr[1])
    neighbors_of_1 = list(pool.neighbor_indices[start:stop])

    assert 1 not in neighbors_of_1, "Distracted human should not perceive robot beyond constricted range (1.5m)"
    assert 2 in neighbors_of_1, "Distracted human should still perceive other humans at normal range"


def test_asymmetric_fov_constriction_for_robot(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    perception = DefaultPerception()

    # Agent 1: Human facing East (theta=0), vision_fov=180 deg, range=5m
    # robot_fov_gain=0.2 -> effective robot FOV is 36 deg (half-angle 18 deg)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.vision_range[0] = 5.0
    pool.vision_fov[0] = 180.0
    pool.proximity_sense[0] = 0.5
    pool.robot_awareness[0] = 1.0
    pool.robot_fov_gain[0] = 0.2

    # Robot at (2.0, 1.5) -> angle ~36.8 deg > 18 deg half-angle (outside robot FOV cone)
    pool.add_agent(agent_factory(agent_id=2, x=2.0, y=1.5))
    pool.kind[1] = KIND_ROBOT

    # Human at (2.0, 1.5) -> within standard 180 deg FOV (half-angle 90 deg)
    pool.add_agent(agent_factory(agent_id=3, x=2.0, y=1.5))
    pool.kind[2] = KIND_HUMAN

    perception.compute_pool(pool)

    start = int(pool.neighbor_indptr[0])
    stop = int(pool.neighbor_indptr[1])
    neighbors_of_1 = list(pool.neighbor_indices[start:stop])

    assert 1 not in neighbors_of_1, "Tunnel-vision human against robot should miss robot outside constricted FOV cone"
    assert 2 in neighbors_of_1, "Human at same position should be perceived by full 180 FOV"


def test_kdtree_robot_awareness_gating(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    # 70 agents (> 64 threshold) triggers _compute_kdtree
    pool = pool_empty(capacity=80)
    perception = DefaultPerception()

    # Agent 0: Human blind to robot
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_awareness[0] = 0.0

    # Agent 1: Robot at (1.0, 0.0)
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT

    # Agents 2..69: Other humans scattered around
    for i in range(2, 70):
        pool.add_agent(agent_factory(agent_id=i + 1, x=float(i % 10) * 0.5, y=float(i // 10) * 0.5))
        pool.kind[i] = KIND_HUMAN

    perception.compute_pool(pool)

    start = int(pool.neighbor_indptr[0])
    stop = int(pool.neighbor_indptr[1])
    neighbors_of_0 = list(pool.neighbor_indices[start:stop])

    assert 1 not in neighbors_of_0, "KD-tree path must gate robot out of blind human neighbor list"
    assert len(neighbors_of_0) > 0, "KD-tree path must preserve nearby humans for blind human"


def test_all_human_crowd_invariant(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    # An all-human crowd should have 100% identical perception graph whether awareness is 1.0 or 0.0
    pool_normal = pool_empty(capacity=16)
    pool_blind = pool_empty(capacity=16)
    perception = DefaultPerception()

    positions = [(0.0, 0.0), (1.0, 0.5), (0.5, 1.0), (2.0, 1.5), (1.5, 2.0)]
    for idx, (x, y) in enumerate(positions):
        pool_normal.add_agent(agent_factory(agent_id=idx + 1, x=x, y=y))
        pool_normal.kind[idx] = KIND_HUMAN
        pool_normal.robot_awareness[idx] = 1.0

        pool_blind.add_agent(agent_factory(agent_id=idx + 1, x=x, y=y))
        pool_blind.kind[idx] = KIND_HUMAN
        pool_blind.robot_awareness[idx] = 0.0

    perception.compute_pool(pool_normal)
    perception.compute_pool(pool_blind)

    np.testing.assert_array_equal(pool_normal.neighbor_indptr, pool_blind.neighbor_indptr)
    np.testing.assert_array_equal(pool_normal.neighbor_indices, pool_blind.neighbor_indices)


def _neighbors(pool: AgentPool, i: int) -> list[int]:
    return list(pool.neighbor_indices[int(pool.neighbor_indptr[i]) : int(pool.neighbor_indptr[i + 1])])


def _add_far_humans(pool: AgentPool, agent_factory: Callable[..., BaseAgent], count: int) -> None:
    for k in range(count):
        pool.add_agent(agent_factory(agent_id=100 + k, x=1000.0 + 10.0 * k, y=1000.0))
        pool.kind[pool.n - 1] = KIND_HUMAN


@pytest.mark.parametrize("far_humans", [0, 70], ids=["dense", "kdtree"])
def test_robot_fov_gain_above_one_widens_the_cone_for_robots_only(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], far_humans: int) -> None:
    pool = pool_empty(capacity=96)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.vision_range[0] = 5.0
    pool.vision_fov[0] = 90.0
    pool.proximity_sense[0] = 0.5
    pool.robot_awareness[0] = 1.0
    pool.robot_fov_gain[0] = 2.0
    pool.add_agent(agent_factory(agent_id=2, x=0.5, y=2.0))
    pool.kind[1] = KIND_ROBOT
    pool.add_agent(agent_factory(agent_id=3, x=0.5, y=2.0))
    pool.kind[2] = KIND_HUMAN
    _add_far_humans(pool, agent_factory, far_humans)

    DefaultPerception().compute_pool(pool)

    assert _neighbors(pool, 0) == [1]
