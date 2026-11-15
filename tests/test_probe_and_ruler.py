import math

import pytest

from blindheads import probe, ruler


def test_midpoint_rule():
    nodes, weights = probe.midpoint_nodes(4)
    assert nodes == [0.125, 0.375, 0.625, 0.875] and weights == [0.25] * 4
    # exact for a linear derivative: the integral of 2 * beta over [0, 1] is 1
    assert math.isclose(probe.integrate([2 * b for b in nodes], weights), 1.0)
    with pytest.raises(ValueError):
        probe.midpoint_nodes(0)


def test_integrate_and_mean_abs_on_nested_lists():
    d = [[[1.0, -2.0]], [[3.0, 4.0]]]
    assert probe.integrate(d, [0.5, 0.5]) == [[2.0, 1.0]]
    # the absolute value is taken per prompt, so gains of opposite sign do not cancel
    assert probe.mean_abs([[[1.0, -2.0]], [[-1.0, 2.0]]]) == [[1.0, 2.0]]
    with pytest.raises(ValueError):
        probe.integrate(d, [1.0])


def test_forfeited_attention():
    assert math.isclose(probe.forfeited_attention(0.9, 0.3, 0.2), 0.12)
    assert probe.forfeited_attention(0.5, 0.5, 0.9) == 0.0          # a head whose selector already finds the needle


def test_needle_token_span():
    text = "x 11: aaaa-bbbb. y 22: cccc-dddd. What is 22?"
    offsets = [(i, i + 1) for i in range(len(text))]                  # one token per character
    s, e = probe.needle_token_span(text, "cccc-dddd", offsets)
    assert text[s:e] == ": cccc-dddd"                                # the lead ": " belongs to the span
    s, e = probe.needle_token_span("zcccc-dddd", "cccc-dddd", [(i, i + 1) for i in range(10)])
    assert (s, e) == (1, 10)                                         # no lead before the answer
    with pytest.raises(ValueError):
        probe.needle_token_span("ab ab", "ab", [(0, 2), (3, 5)])     # the answer must occur once


def test_ruler_metrics():
    assert ruler.string_match_all("the UUID is ABC-1 and def-2.", ["abc-1", "DEF-2"]) == 1.0
    assert ruler.string_match_all("abc-1 only", ["abc-1", "def-2"]) == 0.5
    assert ruler.string_match_part("abc-1 only", ["abc-1", "def-2"]) == 1.0
    assert ruler.accuracy(["abc", "x"], [["abc"], ["y"]]) == 50.0
    assert ruler.prompt_text({"input": "Q", "answer_prefix": " A"}) == "Q A"
    with pytest.raises(ValueError):
        ruler.accuracy([], [])
