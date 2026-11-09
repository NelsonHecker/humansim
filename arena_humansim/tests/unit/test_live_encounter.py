from __future__ import annotations

import numpy as np
import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import (
    AgentConfig,
    ModuleConfig,
    ScenarioConfig,
    SimulationParams,
)
from arena_humansim.utils.types import Pose2D, WaypointMode


def _make_encounter_scenario(planner: str, awareness: float, repulsion: float) -> ScenarioConfig:
    human = AgentConfig(
        agent_id=1,
        agent_type="adult",
        kind=0,
        spawn_pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        goal_sequence=[Pose2D(x=10.0, y=0.0, theta=0.0)],
        waypoint_mode=WaypointMode.ONCE,
        desired_velocity=1.0,
        agent_radius=0.3,
        policy="sfm",
        robot_awareness=awareness,
        robot_repulsion_gain=repulsion,
    )
    robot = AgentConfig(
        agent_id=-1,
        agent_type="adult",
        kind=1,
        spawn_pose=Pose2D(x=6.0, y=0.05, theta=np.pi),
        goal_sequence=[Pose2D(x=0.0, y=0.05, theta=np.pi)],
        waypoint_mode=WaypointMode.ONCE,
        desired_velocity=1.0,
        agent_radius=0.4,
        policy="straight",
    )

    return ScenarioConfig(
        name=f"encounter_{planner}_{int(awareness*10)}_{int(repulsion*10)}",
        simulation=SimulationParams(seed=42, dt=0.05, max_ticks=100),
        modules=ModuleConfig(local_planner=planner),
        agents=[human, robot],
    )


def _simulate_encounter(planner: str, awareness: float, repulsion: float) -> list[tuple[float, float]]:
    from tests.integration._helpers import build_manager
    scenario = _make_encounter_scenario(planner, awareness, repulsion)
    mgr = build_manager(scenario, node_name=f"sim_{planner}_{int(awareness*10)}_{int(repulsion*10)}")
    human_trajectory: list[tuple[float, float]] = []

    idx = mgr._pool.idx(1)
    r_idx = mgr._pool.idx(-1)

    for tick in range(80):
        mgr.tick()
        pool = mgr._pool
        idx = pool.idx(1)
        r_idx = pool.idx(-1)
        human_trajectory.append((float(pool.pos[idx, 0]), float(pool.pos[idx, 1])))

    mgr.destroy_node()
    return human_trajectory


def test_live_head_on_encounter_sfm_blind_vs_aware():
    traj_blind = _simulate_encounter("sfm", awareness=0.0, repulsion=0.0)
    traj_normal = _simulate_encounter("sfm", awareness=1.0, repulsion=1.0)
    traj_scared = _simulate_encounter("sfm", awareness=1.0, repulsion=2.5)

    # Before physical body contact (tick <= 55), blind human has 0 lateral deviation and keeps walking forward
    max_lateral_blind_before_contact = max(abs(y) for _, y in traj_blind[:55])
    assert max_lateral_blind_before_contact < 1e-4, f"Blind human should have 0 lateral deviation, got {max_lateral_blind_before_contact}"
    
    # Aware human yields and decelerates much earlier than blind human:
    # At tick 60, blind human has rushed forward to x=2.56 with velocity ~1.0,
    # whereas normal human has decelerated to vel ~0.59 (x=2.46),
    # and scared human has halted/reversed to vel ~0.15 (x=2.33).
    x_blind_t60 = traj_blind[60][0]
    x_normal_t60 = traj_normal[60][0]
    x_scared_t60 = traj_scared[60][0]

    assert x_blind_t60 > x_normal_t60 > x_scared_t60, (
        f"Blind human ({x_blind_t60:.2f}) rushes furthest; normal human yields ({x_normal_t60:.2f}); scared human yields furthest back ({x_scared_t60:.2f})"
    )

