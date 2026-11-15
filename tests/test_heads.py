import os

import pytest

from blindheads.heads import HEADS_DIR, HeadSet, format_heads, load_heads, parse_heads


def test_parse_and_format():
    assert parse_heads("22.0, 9.0,16.0") == [(9, 0), (16, 0), (22, 0)]
    assert parse_heads(["9.1", (3, 0), "9.1"]) == [(3, 0), (9, 1)]
    assert format_heads([(16, 0), (9, 0)]) == "9.0,16.0"
    for bad in ("9", "a.0", "9.x", "-1.0"):
        with pytest.raises(ValueError):
            parse_heads(bad)


def test_round_trip(tmp_path):
    hs = HeadSet(model="m", selector="s", unit="key-value head", heads=["9.0", "16.0"], total_units=64, scores={"9.0": 0.5})
    path = tmp_path / "x.json"
    hs.save(str(path))
    back = load_heads(str(path))
    assert back.heads == ["9.0", "16.0"] and back.by_layer() == {9: [0], 16: [0]} and back.scores == {"9.0": 0.5}


# the shipped head sets: file name -> (count, units of the checkpoint)
SHIPPED = {"infllm-v2-8b": (3, 64), "minicpm4-8b": (3, 64), "minicpm4.1-8b": (3, 64), "qwen3-8b_seerattention-r": (18, 288),
           "qwen3-4b_seerattention-r": (9, 288), "qwen3-14b_seerattention-r": (5, 320)}


def test_the_shipped_sets():
    shipped = sorted(f[:-5] for f in os.listdir(HEADS_DIR) if f.endswith(".json"))
    assert shipped == sorted(SHIPPED)


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_set(name):
    hs = load_heads(name)
    count, total = SHIPPED[name]
    assert len(hs.heads) == len(set(hs.heads)) == count and hs.total_units == total
    ranked = sorted(hs.scores, key=lambda u: -hs.scores[u])       # the set is the top of its own ranking
    assert len(hs.scores) == total and set(ranked[:count]) == set(hs.heads)


@pytest.mark.parametrize("name", ["infllm-v2-8b", "minicpm4-8b", "minicpm4.1-8b"])
def test_the_lineage_keeps_the_three_blind_heads(name):
    assert load_heads(name).units() == [(9, 0), (16, 0), (22, 0)]


def test_unknown_name():
    with pytest.raises(FileNotFoundError):
        load_heads("no-such-model")
