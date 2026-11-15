import math

import pytest

torch = pytest.importorskip("torch")
from blindheads.infllm_v2 import selection as S  # noqa: E402


def test_quantizer_levels_and_error():
    torch.manual_seed(0)
    keys = torch.randn(4096, 16) * torch.linspace(0.2, 5.0, 16) + torch.linspace(-3, 3, 16)   # channels of different spread
    errors = []
    for bits in (2, 3, 4, 8):
        q = S.quantize_keys(keys, bits)
        assert q.shape == keys.shape
        assert max(len(torch.unique(q[:, c])) for c in range(16)) <= 2 ** bits
        errors.append(float(((q - keys) / keys.std(0)).pow(2).mean().sqrt()))
    assert errors == sorted(errors, reverse=True)            # every extra bit lowers the error
    assert 0.08 < errors[2] < 0.14                            # about 0.11 of the spread at four bits (the paper's figure)
    assert S.quantize_keys(keys, None) is keys


def test_one_bit_is_the_sign_about_the_channel_mean():
    torch.manual_seed(1)
    keys = torch.randn(512, 8) + 2.0
    q = S.quantize_keys(keys, 1)
    mu = keys.mean(0, keepdim=True)
    assert torch.equal(torch.sign(q - mu), torch.sign(keys - mu))
    assert all(len(torch.unique(q[:, c])) <= 2 for c in range(8))


def test_statistics_from_a_prefix_do_not_see_later_keys():
    torch.manual_seed(2)
    keys = torch.randn(1000, 8)
    later = keys.clone(); later[600:] = later[600:] * 7 + 3            # change only the keys after the prefix
    a, b = S.quantize_keys(keys, 4, stats_prefix=500), S.quantize_keys(later, 4, stats_prefix=500)
    assert torch.equal(a[:600], b[:600])                               # the codes of the earlier keys are unchanged
    assert not torch.equal(S.quantize_keys(keys, 4)[:600], S.quantize_keys(later, 4)[:600])


def test_group_mass_sums_to_the_number_of_query_heads():
    torch.manual_seed(3)
    q, k = torch.randn(16, 32), torch.randn(700, 32)
    m = S.group_mass(q, k)
    assert m.shape == (700,) and math.isclose(float(m.sum()), 16.0, rel_tol=1e-4)
    # a key aligned with one query head collects that head's attention
    k[123] = q[5] * 50
    assert int(S.group_mass(q, k).argmax()) == 123


def test_forced_blocks():
    assert S.forced_blocks(position=70, block=64, local_blocks=32, init_blocks=1) == [0, 1]
    assert S.forced_blocks(position=64 * 40 + 3, block=64, local_blocks=2, init_blocks=1) == [0, 38, 39, 40]


def test_fill_selection_keeps_forced_blocks_and_the_budget():
    shipped = torch.full((2, 1, 5), -1, dtype=torch.int32)
    sel = torch.full_like(shipped, -1)
    valid = torch.tensor([True] * 8 + [False] * 2)
    merged = S.fill_selection(sel, 1, order=[9, 7, 0, 3, 6, 2], forced=[0, 5], valid=valid, slots=5, like=shipped)
    assert merged == [0, 5, 7, 3, 6]                 # forced first, block 9 is not valid, block 0 is not repeated
    assert sel[1, 0].tolist() == [0, 3, 5, 6, 7]     # written in block order, as the release writes its selections
    assert sel[0, 0].tolist() == [-1] * 5            # the other head is untouched
    few = torch.full_like(shipped, -1)
    S.fill_selection(few, 0, order=[1], forced=[0], valid=valid, slots=5, like=shipped)
    assert few[0, 0].tolist() == [0, 1, -1, -1, -1]  # fewer candidates than slots: the rest stays empty
