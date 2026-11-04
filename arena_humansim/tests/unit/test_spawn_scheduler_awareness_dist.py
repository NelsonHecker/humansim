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
