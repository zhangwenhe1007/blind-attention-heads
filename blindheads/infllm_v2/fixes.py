"""Training-free fixes for the flagged heads of an InfLLM-V2 checkpoint, installed as hooks on the release's attention.

Every fix leaves the weights, the kernel and the key-value cache as released, and acts only in the key-value heads it is
given. Each `install` function returns a handle whose `remove()` restores the released behaviour and whose `stats`
count what the hook did, so a run can check that the fix was really applied.

    full_attention_heads   the default fix of the paper: the flagged heads read every key
    low_bit_scorer         the flagged heads pick their blocks from a 4-bit copy of the keys, at the shipped budget
    larger_budget          the flagged layers read more blocks, filled from the shipped ranking
    exact_mass             the diagnostic instrument: block selection by exact attention mass (not a deployable path)
"""
import inspect

from ..heads import parse_heads
from .selection import MassSelection


class Handle:
    """What an installed fix returns: `stats` (counters filled as the model runs) and `remove()`."""

    def __init__(self):
        self.stats = {}
        self._undo = []

    def remove(self):
        while self._undo:
            obj, name, value = self._undo.pop()
            setattr(obj, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def _attention_layers(model):
    return [layer.self_attn for layer in model.model.layers]


def full_attention_heads(model, heads):
    """Full attention in the given key-value heads, at every token. Every other head keeps the shipped selection.

    Each layer's `sparse_forward` runs as released. The output of the query heads that share a listed key-value head is
    then replaced by exact causal attention over every key, computed by FlashAttention on the same queries, keys and
    values, which is what the release's dense setting computes.

    heads: "9.0,16.0,22.0", a list of such units, or a HeadSet's `heads`.
    """
    from flash_attn import flash_attn_varlen_func
    units = parse_heads(heads)
    layers = _attention_layers(model)
    handle = Handle()
    stats = handle.stats
    stats.update(calls=0, prefill_calls=0, decode_calls=0, heads=0)
    for li, attn in enumerate(layers):
        hs = sorted(h for (layer, h) in units if layer == li)
        if not hs:
            continue
        if not hasattr(attn, "sparse_forward"):
            raise RuntimeError(f"layer {li} has no sparse_forward")
        if hasattr(attn, "gate_proj"):
            raise NotImplementedError("this attention class mixes a compressed branch into its output (NSA): see research/")
        if max(hs) >= attn.num_key_value_heads:
            raise ValueError(f"layer {li} has {attn.num_key_value_heads} key-value heads, unit {li}.{max(hs)} asked")
        original = attn.sparse_forward
        # some releases pass the rotary table before the lengths: (q, k, v, cis, cu_seqlens_q, cu_seqlens_k, max_q, max_k)
        shift = 1 if "cis" in list(inspect.signature(type(attn).sparse_forward).parameters)[:5] else 0

        def make(original, hs, shift):
            def sparse_forward(q, k, v, *args, **kwargs):
                out = original(q, k, v, *args, **kwargs)
                cu_q, cu_k, max_q, max_k = args[shift:shift + 4]
                group = q.shape[1] // k.shape[1]                 # query heads per key-value head
                decode = int(max_q) == 1
                for h in hs:
                    full = flash_attn_varlen_func(q[:, h * group:(h + 1) * group].contiguous(), k[:, h:h + 1].contiguous(),
                                                  v[:, h:h + 1].contiguous(), cu_q, cu_k, int(max_q), int(max_k),
                                                  dropout_p=0.0, causal=not decode)
                    out[:, h * group:(h + 1) * group] = full.to(out.dtype)
                stats["calls"] += 1
                stats["heads"] += len(hs)
                stats["decode_calls" if decode else "prefill_calls"] += 1
                return out
            return sparse_forward
        attn.sparse_forward = make(original, hs, shift)
        handle._undo.append((attn, "sparse_forward", original))
    if not handle._undo:
        raise ValueError("no listed unit matches a layer of this model")
    return handle


def low_bit_scorer(model, cfg, heads, tokens="decode", bits=4):
    """The 4-bit scorer of Section 6: in the given key-value heads, the free slots of the shipped budget are filled with
    the blocks of largest attention mass under a `bits`-bit copy of the keys. Attention itself reads the full keys.

    tokens: "decode" acts at the generated tokens, "question" also at the last 128 prompt tokens, "every" at every
    token. With "every", the quantizer's statistics come from the first 6,144 keys, so a code never depends on later
    tokens.
    """
    return MassSelection(model, cfg, heads=heads, bits=bits, tokens=tokens, stats_prefix=6144 if tokens == "every" else None)


def exact_mass(model, cfg, heads=None, tokens="decode"):
    """The instrument of Section 5: block selection by exact attention mass at the shipped budget, in the given
    key-value heads (None: every head) and at the given tokens ("decode", "question" or "every"). It reads every key
    to choose the blocks, so it measures what the choice of blocks costs and is not a way to run the model."""
    return MassSelection(model, cfg, heads=heads, bits=None, tokens=tokens)


def larger_budget(model, cfg, layers, factor=4):
    """A larger budget in the given layers at every token: `factor` times the free slots, filled by the shipped
    selector from its own ranking. Needs no new code path, and reads `factor` times the free blocks in those layers."""
    layers = sorted({int(x) for x in layers})
    free = cfg.sparse_config["topk"] - 2              # the free slots: the configured top-k without the initial and own blocks
    handle = Handle()
    handle.stats.update(calls=0, slots=None)
    for li in layers:
        attn = model.model.layers[li].self_attn
        original = attn.sparse_forward
        wide = attn.topk + (int(factor) - 1) * free    # attn.topk is the whole selection: free slots and the local window
        handle.stats["slots"] = wide

        def make(original, attn, wide):
            def sparse_forward(q, k, v, *args, **kwargs):
                kept = attn.topk
                attn.topk = wide
                handle.stats["calls"] += 1
                try:
                    return original(q, k, v, *args, **kwargs)
                finally:
                    attn.topk = kept
            return sparse_forward
        attn.sparse_forward = make(original, attn, wide)
        handle._undo.append((attn, "sparse_forward", original))
    return handle
