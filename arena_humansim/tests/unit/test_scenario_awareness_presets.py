from __future__ import annotations

import yaml

from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim.utils.scenario import AgentConfig, ScenarioConfig, _structure_manual


def test_agent_type_awareness_loading(tmp_path) -> None:
    yaml_content = """
name: test_blind_pedestrian
extends: adult
perception:
  robot_awareness: blind
  robot_repulsion_gain: blind
"""
    file_path = tmp_path / "test_blind.yaml"
    file_path.write_text(yaml_content)

    atype = load_agent_type_from_file(file_path)
    assert atype.perception.robot_awareness.mean == 0.0
    assert atype.perception.robot_repulsion_gain.mean == 0.0


def test_scenario_manual_structure_presets() -> None:
    scenario_dict = {
        "name": "test_awareness_scenario",
        "agents": [
            {
                "agent_id": 1,
                "agent_type": "adult",
                "robot_awareness": "distracted",
                "robot_repulsion_gain": "distracted",
            },
            {
                "agent_id": 2,
                "agent_type": "adult",
                "robot_aware": False,
            },
            {
                "agent_id": 3,
                "agent_type": "adult",
                "robot_awareness": 0.85,
                "robot_repulsion_gain": 1.5,
            },
        ],
        "flow": {
            "sources": [
                {
                    "pose": {"x": 0.0, "y": 0.0},
                    "agent_template": {
                        "agent_type": "adult",
                        "robot_awareness": {"blind": 0.3, "normal": 0.7},
                    },
                }
            ]
        },
    }

    config = _structure_manual(scenario_dict)

    assert len(config.agents) == 3
    assert config.agents[0].robot_awareness == "distracted"
    assert config.agents[1].robot_aware is False
    assert config.agents[2].robot_awareness == 0.85
    assert config.agents[2].robot_repulsion_gain == 1.5

    assert config.flow is not None
    assert len(config.flow.sources) == 1
    assert config.flow.sources[0].agent_template.robot_awareness == {"blind": 0.3, "normal": 0.7}


def test_resolve_awareness_tuple_precedence() -> None:
    from arena_humansim.core.agent_manager import _resolve_awareness_tuple

    # robot_aware: true -> normal
    assert _resolve_awareness_tuple(None, None, None, None, robot_aware=True) == (1.0, 1.0, 1.0, 1.0)
    # robot_aware: false -> blind (reaction defaults to 1.0)
    assert _resolve_awareness_tuple(None, None, None, None, robot_aware=False) == (0.0, 0.0, 0.0, 1.0)

    # explicit repulsion_gain overrides robot_aware default
    assert _resolve_awareness_tuple(None, 2.5, None, None, robot_aware=True) == (1.0, 2.5, 1.0, 1.0)
    assert _resolve_awareness_tuple(None, 1.5, 0.8, 1.2, robot_aware=False) == (0.0, 1.5, 0.8, 1.2)


def test_resolve_awareness_tuple_validation() -> None:
    import pytest
    from arena_humansim.core.agent_manager import _resolve_awareness_tuple

    with pytest.raises(TypeError, match="Scripted agents require a scalar"):
        _resolve_awareness_tuple({"blind": 0.5}, None)

    with pytest.raises(TypeError, match="Scripted agents require a scalar"):
        _resolve_awareness_tuple(1.0, [1.0, 2.0])

    with pytest.raises(ValueError, match="Unknown awareness preset"):
        _resolve_awareness_tuple("nonexistent_preset", None)


def test_agent_type_preset_probe_slots() -> None:
    from arena_humansim.core.agents.types import PerceptionDist

    # Verify field-aware slot indexing for all 4 parameters
    scared_p = PerceptionDist(robot_fov_gain="scared")
    assert scared_p.robot_fov_gain.mean == 1.2

    aggr_p = PerceptionDist(robot_reaction_gain="aggressive")
    assert aggr_p.robot_reaction_gain.mean == 1.2
