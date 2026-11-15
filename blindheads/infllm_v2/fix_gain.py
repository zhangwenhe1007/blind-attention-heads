"""The calibration probe's fix gain on an InfLLM-V2 checkpoint (Section 5 of the paper, Equations 4 and 5).

Every key-value head u gets a gate alpha_u that blends, at every prompt and answer position, the output of its query
heads over the shipped selection with their output over every key,

    o(alpha) = o_ship + alpha_u (o_full - o_ship),

so alpha = 0 is the shipped model and alpha = 1 is full attention in every head. F_x(alpha) is the log-probability of the
needle tokens of calibration prompt x, read from one teacher-forced pass over the prompt and the needle tokens. One
backward pass gives dF_x / dalpha_u for every head at once, as the sum over positions and the head's query heads of
grad(o) . (o_full - o_ship). The integrated fix gain of a prompt is the integral of that derivative along
alpha = beta * 1 for beta from 0 to 1, by the midpoint rule, and the score of a head is the mean over prompts of its
absolute value.

The selections are recomputed at every point of the path and are constants of the derivative. Memory stays bounded by
gradient checkpointing per decoder layer: the hook that reads the derivative fires while its layer is recomputed, so the
difference o_full - o_ship of one layer is alive at a time.

Cost: m + 2 forward passes and m + 1 backward passes over each prompt for m midpoint nodes (the paper uses m = 16).
"""
import inspect
import time

import torch

from .. import probe as P
from .. import ruler


class FixGainProbe:
    """Installs the gates on a loaded sparse model. Use as a context manager, or call `remove()` when done."""

    def __init__(self, model, tok, cfg):
        from flash_attn import flash_attn_varlen_func
        self.model, self.tok, self.cfg = model, tok, cfg
        self.layers = cfg.num_hidden_layers
        self.query_heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.state = {"on": False, "beta": 0.0, "acc": None, "calls": 0}
        self._undo = []
        attn0 = model.model.layers[0].self_attn
        if hasattr(attn0, "gate_proj"):
            raise NotImplementedError("this attention class mixes a compressed branch into its output (NSA): see research/")
        shift = 1 if "cis" in list(inspect.signature(type(attn0).sparse_forward).parameters)[:5] else 0
        st = self.state

        def blend(li, o_ship, o_full):
            beta = st["beta"]
            out = o_ship if beta == 0.0 else o_ship + beta * (o_full - o_ship)
            if out.requires_grad:                    # under checkpointing only the recomputation builds the graph
                delta = (o_full - o_ship).detach()

                def hook(g, li=li, delta=delta):
                    d = (g.float() * delta.float()).sum(-1)                   # [positions, query heads]
                    st["acc"][li] += d.reshape(-1, d.shape[-1]).sum(0).double()
                    st["calls"] += 1
                out.register_hook(hook)
            return out

        for li, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            original = attn.sparse_forward

            def make(original, li):
                def sparse_forward(q, k, v, *args, **kwargs):
                    o_ship = original(q, k, v, *args, **kwargs)
                    if not st["on"] or q.shape[0] == 1:      # a decode step runs as released
                        return o_ship
                    cu_q, cu_k, max_q, max_k = args[shift:shift + 4]
                    with torch.set_grad_enabled(torch.is_grad_enabled() and st["beta"] > 0.0):
                        o_full = flash_attn_varlen_func(q, k, v, cu_q, cu_k, int(max_q), int(max_k), dropout_p=0.0,
                                                        causal=True)
                    return blend(li, o_ship, o_full.to(o_ship.dtype))
                return sparse_forward
            attn.sparse_forward = make(original, li)
            self._undo.append((attn, "sparse_forward", original))
        self._grad_flags = [(p, p.requires_grad) for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        # the input embeddings start the graph, so that checkpointed layers have something to differentiate through
        self._embed_hook = model.model.embed_tokens.register_forward_hook(
            lambda m, i, o: o.detach().requires_grad_(True) if st["on"] and torch.is_grad_enabled() else o)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        # the release's output head divides the hidden states before the vocabulary projection
        source = inspect.getsource(type(model).forward)
        if getattr(cfg, "pretraining_tp", 1) > 1:
            raise NotImplementedError("pretraining_tp > 1")
        self.head_scale = cfg.hidden_size / cfg.dim_model_base if "dim_model_base" in source else 1.0

    def remove(self):
        while self._undo:
            obj, name, value = self._undo.pop()
            setattr(obj, name, value)
        self._embed_hook.remove()
        self.model.gradient_checkpointing_disable()
        for p, flag in self._grad_flags:
            p.requires_grad_(flag)
        self.model.eval()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def _needle_logits(self, seq, mask, k):
        h = self.model.model(input_ids=seq, attention_mask=mask, use_cache=False)[0]
        return self.model.lm_head(h[0, -k:] / self.head_scale).float()

    def _log_prob(self, seq, mask, target, beta, grad):
        """F_x at alpha = beta * 1 and, with grad, its derivative per (layer, query head)."""
        st = self.state
        st.update(on=True, beta=beta, acc=torch.zeros(self.layers, self.query_heads, dtype=torch.float64, device=seq.device))
        try:
            with torch.set_grad_enabled(grad):
                lp = torch.log_softmax(self._needle_logits(seq, mask, len(target)), dim=-1)[torch.arange(len(target)), target]
                total = lp.sum()
                if grad:
                    total.backward()
        finally:
            st["on"] = False
        return float(total), lp.detach().tolist(), (st["acc"].cpu() if grad else None)

    def prompt(self, row, steps=10, nodes=16):
        """The read-out of one calibration prompt (a RULER row with one needle that the answer copies).

        Returns {"gain": [layers][key-value heads] integrated fix gain, "first_order": the derivative at alpha = 0,
        "log_prob_shipped", "log_prob_full", "completeness": {"sum": sum of the gains, "gap": F(1) - F(0)}}.
        """
        text = ruler.prompt_text(row)
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True)
        ids = enc.input_ids.to(self.model.device)
        start, end = P.needle_token_span(text, row["outputs"][0], enc.offset_mapping[0].tolist())
        needle = ids[0, start:end][:steps].tolist()
        seq = torch.cat([ids, torch.tensor([needle[:-1]], device=ids.device, dtype=ids.dtype)], dim=1)
        target = torch.tensor(needle, device=ids.device)
        mask = torch.ones_like(seq)
        group = self.query_heads // self.kv_heads
        per_unit = lambda g: g.view(self.layers, self.kv_heads, group).sum(-1)      # noqa: E731
        betas, weights = P.midpoint_nodes(nodes)
        self.model.train()                           # gradient checkpointing is active in training mode only
        f0, logp0, g0 = self._log_prob(seq, mask, target, 0.0, True)
        derivs = []
        for beta in betas:
            _, _, g = self._log_prob(seq, mask, target, beta, True)
            derivs.append(per_unit(g))
        f1, _, _ = self._log_prob(seq, mask, target, 1.0, False)
        gain = sum(w * d for w, d in zip(weights, derivs))
        torch.cuda.empty_cache()
        return {"prompt_tokens": int(ids.shape[1]), "needle_tokens": len(needle), "gain": gain.tolist(),
                "first_order": per_unit(g0).tolist(), "log_prob_shipped": f0, "log_prob_full": f1,
                "log_prob_tokens_shipped": logp0, "completeness": {"sum": float(gain.sum()), "gap": f1 - f0},
                "_ids": ids, "_needle": needle}

    @torch.no_grad()
    def decode_check(self, record):
        """The largest gap, in nats per needle token, between the one-pass log-probabilities of a prompt and the
        release's own step-by-step decode of the same needle tokens. A self-check of the teacher-forced pass."""
        from transformers.cache_utils import DynamicCache
        ids, needle = record["_ids"], record["_needle"]
        self.model.eval()
        out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=DynamicCache(), use_cache=True)
        cache, mask = out.past_key_values, torch.ones_like(ids)
        dec = [float(torch.log_softmax(out.logits[0, -1].float(), dim=-1)[needle[0]])]
        for j in range(len(needle) - 1):
            mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
            out = self.model(input_ids=torch.tensor([[needle[j]]], device=ids.device), attention_mask=mask,
                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            dec.append(float(torch.log_softmax(out.logits[0, -1].float(), dim=-1)[needle[j + 1]]))
        return max(abs(a - b) for a, b in zip(dec, record["log_prob_tokens_shipped"]))


def integrated_fix_gain(model, tok, cfg, rows, steps=10, nodes=16, check_decode=2, log=print):
    """Run the probe over calibration prompts.

    Returns {"scores": {"LAYER.HEAD": score}, "ranking": units by decreasing score, "per_prompt": [...],
    "completeness": mean |sum of gains - gap| and mean |gap| over the prompts, "decode_check": [...]}.
    """
    t0 = time.time()
    per_prompt, checks = [], []
    with FixGainProbe(model, tok, cfg) as probe:
        for i, row in enumerate(rows):
            rec = probe.prompt(row, steps=steps, nodes=nodes)
            if i < check_decode:
                checks.append(probe.decode_check(rec))
            rec.pop("_ids"); rec.pop("_needle")
            per_prompt.append(rec)
            if log is not None:
                c = rec["completeness"]
                log(f"  [{i + 1}/{len(rows)}] log p shipped {rec['log_prob_shipped']:.3f}, full {rec['log_prob_full']:.3f}, "
                    f"sum of gains {c['sum']:.3f} against gap {c['gap']:.3f}, {time.time() - t0:.0f} s")
        if probe.state["calls"] == 0:
            raise RuntimeError("the derivative hook never fired")
    mean = P.mean_abs([r["gain"] for r in per_prompt])
    scores = {f"{layer}.{h}": mean[layer][h] for layer in range(len(mean)) for h in range(len(mean[layer]))}
    n = len(per_prompt)
    return {"scores": scores, "ranking": sorted(scores, key=lambda u: -scores[u]), "per_prompt": per_prompt,
            "nodes": nodes, "steps": steps,
            "completeness": {"mean_abs_residual": sum(abs(r["completeness"]["sum"] - r["completeness"]["gap"]) for r in per_prompt) / n,
                             "mean_abs_gap": sum(abs(r["completeness"]["gap"]) for r in per_prompt) / n},
            "decode_check": checks, "seconds": time.time() - t0}
