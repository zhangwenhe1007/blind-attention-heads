"""The threshold rule (Section 5 of the paper): from a score per selection unit to the set of flagged units.

The flagged units are those above a threshold, H(tau) = {u : score_u >= tau}. The threshold starts at the largest score
and is lowered by a constant factor per step, sqrt(2) by default. Each step that adds units is tried on the calibration
prompts with the fix applied to its units, and the rule keeps the first threshold whose units close a share rho of the
gap between the shipped selector and full attention in every unit, 0.9 by default.

Nothing here depends on a model. `threshold_steps` lists the candidate sets, and `threshold_rule` walks them with a
callback that measures the calibration accuracy of a set.
"""
import math

SHARE = 0.9
STEP = math.sqrt(2.0)


def threshold_steps(scores, step=STEP, max_units=None):
    """The sweep's candidate sets.

    scores: {unit: score}, scores non-negative. Returns [(index, tau, [units by decreasing score])] for the thresholds
    tau_i = max(scores) / step**i, one entry per step whose set is larger than the previous one, until every unit with a
    positive score is in (or the set would exceed max_units).
    """
    if step <= 1.0:
        raise ValueError("the step must be larger than 1")
    ranked = sorted(scores, key=lambda u: (-scores[u], str(u)))
    if not ranked or scores[ranked[0]] <= 0:
        return []
    top, positive = scores[ranked[0]], sum(1 for u in ranked if scores[u] > 0)
    out, last, i = [], 0, 0
    while last < positive:
        tau = top / step ** i
        k = sum(1 for u in ranked if scores[u] >= tau)
        if max_units is not None and k > max_units:
            break
        if k > last:
            out.append((i, tau, ranked[:k]))
            last = k
        i += 1
        if i > 4096:                          # scores that span more than 4096 steps of the grid
            break
    return out


def gap_share(value, shipped, full):
    """The share of the gap between the shipped selector and full attention in every unit that `value` closes."""
    if full == shipped:
        raise ValueError("the shipped selector and full attention score the same: there is no gap to close")
    return (value - shipped) / (full - shipped)


def threshold_rule(scores, evaluate, shipped, full, share=SHARE, step=STEP, max_units=None):
    """Walk the sweep and keep the first set that closes `share` of the gap.

    evaluate(units) -> the calibration accuracy with the fix in `units` (a list, by decreasing score).
    shipped, full: the calibration accuracy of the shipped selector and of full attention in every unit.
    Returns {"units": the kept set or None when no step reaches the share, "tau", "share", "curve": [(count, share)]}.
    """
    curve = []
    for _, tau, units in threshold_steps(scores, step, max_units):
        s = gap_share(evaluate(units), shipped, full)
        curve.append((len(units), s))
        if s >= share:
            return {"units": units, "tau": tau, "share": s, "curve": curve}
    return {"units": None, "tau": None, "share": None, "curve": curve}
