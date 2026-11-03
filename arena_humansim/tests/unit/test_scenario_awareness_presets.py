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
