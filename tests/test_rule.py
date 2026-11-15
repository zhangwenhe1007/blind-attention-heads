import math

import pytest

from blindheads import rule
from blindheads.heads import load_heads


def test_steps_on_a_small_example():
    scores = {"a": 8.0, "b": 6.0, "c": 4.1, "d": 3.9, "e": 0.9, "f": 0.0}
    steps = rule.threshold_steps(scores, step=2.0)
    # tau = 8, 4, 2, 1, 0.5: the sets grow to 1, 3, 4 and 5 units, the step at tau = 1 adds none and is skipped
    assert [(i, len(u)) for i, _, u in steps] == [(0, 1), (1, 3), (2, 4), (4, 5)]
    assert steps[1][2] == ["a", "b", "c"] and math.isclose(steps[1][1], 4.0)
    assert rule.threshold_steps({}) == [] and rule.threshold_steps({"a": 0.0}) == []
    assert [len(u) for _, _, u in rule.threshold_steps(scores, step=2.0, max_units=3)] == [1, 3]
    with pytest.raises(ValueError):
        rule.threshold_steps(scores, step=1.0)


def test_rule_keeps_the_first_step_that_closes_the_share():
    scores = {"a": 8.0, "b": 6.0, "c": 4.1, "d": 3.9, "e": 0.9}
    acc = {1: 40.0, 3: 85.0, 4: 97.0, 5: 99.0}
    tried = []

    def evaluate(units):
        tried.append(len(units))
        return acc[len(units)]

    kept = rule.threshold_rule(scores, evaluate, shipped=30.0, full=100.0, share=0.9, step=2.0)
    assert kept["units"] == ["a", "b", "c", "d"] and tried == [1, 3, 4]        # the last step is never run
    assert math.isclose(kept["share"], 67.0 / 70.0)
    none = rule.threshold_rule(scores, lambda units: 50.0, shipped=30.0, full=100.0, step=2.0)
    assert none["units"] is None and len(none["curve"]) == 4
    with pytest.raises(ValueError):
        rule.gap_share(1.0, 5.0, 5.0)


RULE_SETS = ["qwen3-8b_seerattention-r", "qwen3-4b_seerattention-r", "qwen3-14b_seerattention-r"]


@pytest.mark.parametrize("name", RULE_SETS)
def test_the_rule_reproduces_the_paper_from_the_shipped_scores(name):
    """Scores and calibration curve of the paper in, the paper's head set out."""
    hs = load_heads(name)
    cal = hs.calibration
    shipped, full = cal["accuracy_shipped"], cal["accuracy_full_attention"]
    curve = {count: share for count, share in cal["curve"]}
    counts = [len(u) for _, _, u in rule.threshold_steps(hs.scores)]
    assert counts[:len(curve)] == sorted(curve)                                 # the sweep visits the counts that were run
    kept = rule.threshold_rule(hs.scores, lambda units: shipped + curve[len(units)] * (full - shipped), shipped, full)
    assert kept["units"] is not None and set(kept["units"]) == set(hs.heads)
    assert kept["share"] >= 0.9 and all(s < 0.9 for _, s in kept["curve"][:-1])
