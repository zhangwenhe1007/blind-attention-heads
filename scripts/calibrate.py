#!/usr/bin/env python
"""Calibrate an InfLLM-V2 checkpoint: probe every key-value head, then let the threshold rule pick the flagged heads.

    python scripts/calibrate.py --model MODEL --prompts calibration.jsonl --out my_heads.json

Step 1, the probe.  The integrated fix gain of every key-value head on the calibration prompts (Equation 5).
Step 2, the rule.   The threshold on the score is lowered by sqrt(2) per step. Each step that adds heads is run on the
                    calibration prompts with full attention in its heads, and the first step that closes --share of the gap
                    between the shipped selector and full attention is kept.

Calibration prompts must come from a generator seed that no evaluation uses, and must contain look-alike competitors
(RULER's multi-key task does; a single needle among filler sentences flags no head at all).

Cost on one GPU at 32K: the probe takes a few minutes per prompt at 16 midpoint nodes, and each step of the rule runs
every calibration prompt once. --probe_prompts and --rule_prompts trade cost for stability. The paper uses 60 probe
prompts on this lineage and 200 on the other selectors.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blindheads import HeadSet, infllm_v2, ruler, rule  # noqa: E402
from blindheads.infllm_v2.fix_gain import integrated_fix_gain  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="calibration prompts (RULER multi-key, a seed no evaluation uses)")
    ap.add_argument("--out", required=True, help="the head file to write")
    ap.add_argument("--probe_prompts", type=int, default=20)
    ap.add_argument("--rule_prompts", type=int, default=50)
    ap.add_argument("--nodes", type=int, default=16, help="midpoint nodes of the integral")
    ap.add_argument("--steps", type=int, default=10, help="needle tokens teacher-forced per prompt")
    ap.add_argument("--share", type=float, default=rule.SHARE)
    ap.add_argument("--max_heads", type=int, default=None, help="stop the sweep beyond this many heads")
    ap.add_argument("--scores", default=None, help="reuse the scores of an earlier run (its --out file, its .scores.json file "
                                                   "or a shipped head file) and skip the probe")
    ap.add_argument("--name", default=None, help="the checkpoint's name in the head file (default: --model)")
    a = ap.parse_args()
    model, tok, cfg = infllm_v2.load(a.model)
    total = cfg.num_hidden_layers * cfg.num_key_value_heads
    if a.scores:
        scores, probe_info = json.load(open(a.scores))["scores"], {"reused": os.path.abspath(a.scores)}
    else:
        print(f"probe: integrated fix gain on {a.probe_prompts} prompts, {a.nodes} nodes")
        pr = integrated_fix_gain(model, tok, cfg, ruler.read_prompts(a.prompts, a.probe_prompts), a.steps, a.nodes)
        scores = pr["scores"]
        probe_info = {"prompts": a.probe_prompts, "nodes": a.nodes, "completeness": pr["completeness"],
                      "decode_check": pr["decode_check"], "seconds": round(pr["seconds"])}
        print("top heads by fix gain: " + ", ".join(f"{u} ({scores[u]:.3f})" for u in pr["ranking"][:8]))
        kept_scores = os.path.splitext(a.out)[0] + ".scores.json"          # the probe is the expensive step: keep its result
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(kept_scores, "w") as f:
            json.dump({"scores": scores, "probe": probe_info}, f, indent=1)
        print(f"probe scores saved to {kept_scores} (pass --scores {kept_scores} to rerun the rule without the probe)")
    rows = ruler.read_prompts(a.prompts, a.rule_prompts)
    print(f"rule: the shipped selector on {len(rows)} calibration prompts")
    shipped = infllm_v2.evaluate(model, tok, rows)["accuracy"]
    print(f"rule: full attention in every head ({total} heads)")
    every = [f"{layer}.{h}" for layer in range(cfg.num_hidden_layers) for h in range(cfg.num_key_value_heads)]
    with infllm_v2.full_attention_heads(model, every):
        full = infllm_v2.evaluate(model, tok, rows)["accuracy"]
    print(f"      shipped {shipped:.1f}, full attention {full:.1f}")
    if full <= shipped:
        raise SystemExit(f"the shipped selector scores {shipped:.1f} and full attention {full:.1f} on these {len(rows)} prompts: there is "
                         "no gap for the rule to close. Use more prompts (--rule_prompts), or prompts on which the selector loses.")

    def run(units):
        with infllm_v2.full_attention_heads(model, units):
            acc = infllm_v2.evaluate(model, tok, rows)["accuracy"]
        print(f"      {len(units):3d} heads: {acc:.1f}  (share of the gap {rule.gap_share(acc, shipped, full):.2f})")
        return acc

    kept = rule.threshold_rule(scores, run, shipped, full, share=a.share, max_units=a.max_heads)
    if kept["units"] is None:
        raise SystemExit(f"no step of the sweep closes {a.share} of the gap (heads, share): {kept['curve']}. "
                         "Raise --max_heads, or use more prompts for the probe.")
    HeadSet(model=a.name or a.model, selector="InfLLM-V2", unit="key-value head", heads=kept["units"], total_units=total,
            source="scripts/calibrate.py: the threshold rule on the integrated fix gain", scores=scores,
            calibration={"prompts_file": os.path.abspath(a.prompts), "probe": probe_info, "rule_prompts": len(rows),
                         "share": a.share, "step": "sqrt(2)", "accuracy_shipped": shipped, "accuracy_full_attention": full,
                         "curve": kept["curve"], "share_at_kept_count": kept["share"]}).save(a.out)
    print(f"flagged heads: {', '.join(kept['units'])}  ->  {a.out}")


if __name__ == "__main__":
    main()
