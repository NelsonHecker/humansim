from __future__ import annotations

import numpy as np

from arena_humansim.core.spawn_scheduler import sample_awareness_spec, sample_repulsion_gain_spec


def test_sample_awareness_presets() -> None:
    rng = np.random.default_rng(42)

    assert sample_awareness_spec("blind", rng) == 0.0
    assert sample_repulsion_gain_spec("blind", rng) == 0.0

    assert sample_awareness_spec("distracted", rng) == 0.3
    assert sample_repulsion_gain_spec("distracted", rng) == 0.6

    assert sample_awareness_spec("normal", rng) == 1.0
    assert sample_repulsion_gain_spec("normal", rng) == 1.0

    assert sample_awareness_spec("scared", rng) == 1.0
    assert sample_repulsion_gain_spec("scared", rng) == 2.0

    assert sample_awareness_spec("aggressive", rng) == 1.0
    assert sample_repulsion_gain_spec("aggressive", rng) == 0.4

    assert sample_awareness_spec(True, rng) == 1.0
    assert sample_awareness_spec(False, rng) == 0.0


def test_sample_awareness_discrete_mixture() -> None:
    rng = np.random.default_rng(123)
    spec = {"blind": 0.4, "normal": 0.6}

    samples = [sample_awareness_spec(spec, rng) for _ in range(1000)]
    blind_count = sum(1 for s in samples if s == 0.0)
    normal_count = sum(1 for s in samples if s == 1.0)

    # 40% blind, 60% normal within standard confidence bound
    assert 350 <= blind_count <= 450
    assert 550 <= normal_count <= 650


def test_sample_awareness_continuous_distribution() -> None:
    rng = np.random.default_rng(456)
    spec = {"mean": 0.7, "std": 0.1, "clip_low": 0.2, "clip_high": 0.9}

    samples = np.array([sample_awareness_spec(spec, rng) for _ in range(1000)])

    assert np.all(samples >= 0.2)
    assert np.all(samples <= 0.9)
    assert abs(float(samples.mean()) - 0.7) < 0.02


def test_sample_awareness_validation() -> None:
    import pytest

    rng = np.random.default_rng(789)

    with pytest.raises(ValueError, match="Unknown awareness preset"):
        sample_awareness_spec("invalid_preset", rng)

    with pytest.raises(ValueError, match="Mixture weights must be non-negative"):
        sample_awareness_spec({"blind": -0.5, "normal": 0.5}, rng)

    # List of names without weights dicts
    list_spec = ["blind", "normal"]
    res = sample_awareness_spec(list_spec, rng)
    assert res in (0.0, 1.0)


def test_spawn_scheduler_all_four_fields_propagation() -> None:
    from arena_humansim.core.spawn_scheduler import SpawnScheduler
    from arena_humansim.utils.types import AgentTemplate, Pose2D, Shape, SourceConfig

    scheduler = SpawnScheduler(rng=np.random.default_rng(42))
    source = SourceConfig(
        name="source_0",
        pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        shape=Shape(type="circle", radius=1.0),
        agent=AgentTemplate(
            agent_type="adult",
            robot_awareness=0.5,
            robot_repulsion_gain=1.8,
            robot_fov_gain=0.7,
            robot_reaction_gain=1.3,
        ),
    )
    scheduler.add_source(source)
    # Trigger sample
    req = scheduler._sample_spawn_request(source, tick_count=0, seq=1)
    assert req.robot_awareness == 0.5
    assert req.robot_repulsion_gain == 1.8
    assert req.robot_fov_gain == 0.7
    assert req.robot_reaction_gain == 1.3
