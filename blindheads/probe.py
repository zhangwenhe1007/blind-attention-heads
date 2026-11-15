"""The calibration probe's two read-outs, as arithmetic on what a model integration records (Section 5 of the paper).

Forfeited attention.  F_u = (r_u(exact) - r_u(shipped)) * m_u: the needle recall that the shipped selector loses against
exact mass selection at the same budget, times the attention the unit puts on the needle block. It explains why a unit
fails.

Fix gain.  Each unit u has a gate alpha_u that blends, at every position where the selector acts, the output of its
query heads over the shipped selection with their output over every key, o(alpha) = o_ship + alpha_u (o_full - o_ship).
With F_x(alpha) the log-probability of the needle tokens of calibration prompt x, the fix gain of u is the integral of
dF_x / dalpha_u along alpha = beta * 1 for beta from 0 to 1, in absolute value per prompt, averaged over the prompts.
The integral is evaluated by the midpoint rule on m points, each one forward and one backward pass.
"""


def midpoint_nodes(m):
    """Nodes and weights of the midpoint rule on [0, 1]: m points (k + 1/2) / m with weight 1 / m."""
    if m < 1:
        raise ValueError("the midpoint rule needs at least one point")
    return [(k + 0.5) / m for k in range(m)], [1.0 / m] * m


def integrate(derivatives, weights):
    """sum_k w_k * derivatives[k], element by element: the integrated gradient of one prompt from its node derivatives.

    derivatives: one nested list (or tensor) per node, all of one shape. Returns the same shape.
    """
    if len(derivatives) != len(weights):
        raise ValueError("one derivative per node")
    total = None
    for d, w in zip(derivatives, weights):
        total = _scale(d, w) if total is None else _add(total, _scale(d, w))
    return total


def mean_abs(per_prompt):
    """The score of every unit: the mean over prompts of the absolute value of its per-prompt gain."""
    if not per_prompt:
        raise ValueError("no prompt")
    total = None
    for g in per_prompt:
        a = _map(abs, g)
        total = a if total is None else _add(total, a)
    return _scale(total, 1.0 / len(per_prompt))


def forfeited_attention(recall_exact, recall_shipped, needle_mass):
    """F_u from the three quantities the probe records for a unit, each a mean over the probed answer tokens."""
    return (recall_exact - recall_shipped) * needle_mass


def needle_token_span(text, answer, offsets, lead=": "):
    """Token span [start, end) of `lead + answer` inside `text`, from the tokenizer's (char_start, char_end) offsets.

    The answer must occur exactly once. When it is not preceded by `lead`, the span covers the answer alone. A token that
    straddles the first character of the span is included.
    """
    if text.count(answer) != 1:
        raise ValueError(f"the answer occurs {text.count(answer)} times in the text")
    a0 = text.index(answer)
    c0 = a0 - len(lead) if text[max(0, a0 - len(lead)):a0] == lead else a0
    c1 = a0 + len(answer)
    toks = [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1 and e > s]
    if not toks:
        raise ValueError("no token overlaps the needle")
    return toks[0], toks[-1] + 1


def _map(fn, x):
    return [_map(fn, y) for y in x] if isinstance(x, (list, tuple)) else fn(x)


def _scale(x, w):
    return [_scale(y, w) for y in x] if isinstance(x, (list, tuple)) else x * w


def _add(a, b):
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("shapes differ")
        return [_add(x, y) for x, y in zip(a, b)]
    return a + b
