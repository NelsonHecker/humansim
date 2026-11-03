from __future__ import annotations

from collections.abc import Callable

import numpy as np

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
