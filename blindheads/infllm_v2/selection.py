"""Block selection by attention mass on the InfLLM-V2 release: the paper's diagnostic instrument and its 4-bit scorer.

InfLLM-V2 picks the blocks of every query in `compressed_attention`, from mean-pooled keys. The hook installed here
replaces that pick, in the key-value heads and at the tokens it is given, by the blocks of largest attention mass,

    M_b = sum over the query heads of the group of  sum_{j in block b} softmax_j(q . k_j / sqrt(d)),

computed on the exact keys (Equation 2, the instrument of Section 5) or on a low-bit copy of them (Equation 7, the
scorer of Section 6). The budget and the blocks the release always reads (the initial block, the local window and the
query's own block) are kept: only the free slots are refilled. Every other head and token keeps the shipped selection.

The functions below the hook are the pieces it is built from: the quantizer of Equation 6, the mass of a group, the
release's own block scores (to learn which blocks it forces), and the filling of a selection.
"""
import inspect
import math
import sys

import torch

from ..heads import parse_heads

# the step of the uniform quantizer that is optimal for a unit Gaussian, per bit width (Max, 1960)
UNIFORM_STEP = {2: 0.996, 3: 0.586, 4: 0.335, 8: 0.0308}
TOKENS = {"decode": 0, "question": 128, "every": 100000}       # prompt tokens whose selection is rewritten, from the end


def quantize_keys(keys, bits, stats_prefix=None):
    """A low-bit copy of the keys of one head, used only to score blocks (Equation 6). keys [T, D].

    Each channel is standardized by its mean and standard deviation and quantized uniformly to 2**bits levels with the
    Gaussian-optimal step. One bit keeps the sign about the channel mean. With stats_prefix the statistics come from
    the first stats_prefix keys only, so the code of a key never depends on later keys. bits None returns the keys.
    """
    if bits is None:
        return keys
    ref = keys if stats_prefix is None else keys[:max(int(stats_prefix), 2)]
    mu = ref.mean(0, keepdim=True)
    dev = keys - mu
    if bits == 1:
        return mu + torch.sign(dev) * (ref - mu).abs().mean(0, keepdim=True)
    sd = (ref - mu).pow(2).mean(0, keepdim=True).sqrt().clamp(min=1e-8)
    step, half = UNIFORM_STEP[bits], 2 ** (bits - 1) - 0.5
    z = torch.clamp(torch.floor(dev / (sd * step)) + 0.5, -half, half)
    return mu + z * step * sd


def group_mass(queries, keys):
    """Attention mass of every key under the query heads of one group, summed over them. queries [G, D], keys [T, D]."""
    return torch.softmax((queries @ keys.T) / math.sqrt(queries.shape[-1]), dim=-1).sum(0)


def forced_blocks(position, block, local_blocks, init_blocks):
    """The blocks the release always reads for a query at a token position."""
    qb = position // block
    return sorted(set(range(init_blocks)) | set(range(max(0, qb - local_blocks), qb + 1)))


def fill_selection(selection, head, order, forced, valid, slots, like, row=0):
    """Write one head's selection: the forced blocks, then `order` (valid blocks only, no repeats) up to `slots`."""
    merged, seen = list(forced), set(forced)
    for b in order:
        if len(merged) >= slots:
            break
        if b in seen or (valid is not None and not bool(valid[b])):
            continue
        merged.append(b); seen.add(b)
    idx = torch.tensor(sorted(merged), device=selection.device).to(like.dtype)
    selection[head, row, :idx.numel()] = idx
    return merged


def _decode_arguments(signature, args, kwargs):
    """The arguments of an intercepted `compressed_attention` call at a decode step, or None at prefill."""
    bound = signature.bind(*args, **kwargs); bound.apply_defaults()
    A = bound.arguments
    cache_lens = A.get("cache_lens")
    if cache_lens is None or bool((cache_lens == 0).all()):
        return None
    if "k2" not in A:
        raise RuntimeError(f"unknown compressed_attention signature: {list(A)}")
    return A


def _release_block_scores(module, A):
    """The release's own block scores at a decode step, with its +inf on the blocks it forces. [key-value heads, blocks]."""
    q_idx = A["cache_lens"] // A["block_size"]
    score = module.infllmv2_attn_stage1(A["q"].contiguous(), A["k"].contiguous(), A["k2"].contiguous(),
                                        cu_seqlens_q=A["cu_seqlens_q"], cu_seqlens_k=A["cu_seqlens_k"],
                                        cu_seqlens_v=A["cu_seqlens_k2"], max_seqlen_q=A["max_seqlen_q"],
                                        max_seqlen_k=A["max_seqlen_k"], causal=False)
    score = score[:, :q_idx.shape[0], :]
    pooled = module.max_pooling_1d_varlen(score.contiguous(), A["cu_seqlens_q"], A["cu_seqlens_k"], A["cache_lens"],
                                          A["max_seqlen_q"], A["max_seqlen_k"], local_blocks=A["local_blocks"],
                                          init_blocks=A["init_blocks"], block_size=A["block_size"], stride=A["kernel_stride"])
    return pooled[:, 0, :].float(), int(q_idx[0])


def _forced_and_valid(scores, n_blocks, q_idx):
    """From one head's release scores: the blocks it forces (+inf) and the blocks a query may select (up to its own)."""
    n = min(n_blocks, scores.shape[0])
    s = scores[:n]
    valid = torch.arange(n, device=s.device) <= q_idx
    forced = torch.nonzero(torch.isinf(s) & (s > 0) & valid).flatten().tolist()
    return forced, valid


class MassSelection:
    """The installed hook. `stats` counts what it did, `remove()` restores the release."""

    def __init__(self, model, cfg, heads=None, bits=None, tokens="decode", stats_prefix=None):
        if tokens not in TOKENS:
            raise ValueError(f"tokens must be one of {sorted(TOKENS)}")
        self.model, self.cfg = model, cfg
        self.block = cfg.sparse_config["block_size"]
        self.units = None if heads is None else set(parse_heads(heads))
        self.layers = None if self.units is None else {layer for layer, _ in self.units}
        self.bits, self.tail, self.stats_prefix = bits, TOKENS[tokens], stats_prefix
        self.stats = {"decode_calls": 0, "decode_heads": 0, "prefill_tokens": 0, "forced_blocks_missing": 0, "errors": []}
        self.module = sys.modules[type(model).__module__]
        self._stash = {}
        self._undo = []
        self._original = self.module.compressed_attention
        self._signature = inspect.signature(self._original)
        for li, layer in enumerate(model.model.layers):       # remember each layer's queries and keys for the selector
            attn = layer.self_attn
            original = attn.sparse_forward

            def make(original, li):
                def sparse_forward(q, k, v, *args, **kwargs):
                    self._stash["qk"], self._stash["layer"] = (q, k), li
                    return original(q, k, v, *args, **kwargs)
                return sparse_forward
            attn.sparse_forward = make(original, li)
            self._undo.append((attn, "sparse_forward", original))
        self.module.compressed_attention = self._select
        self._undo.append((self.module, "compressed_attention", self._original))

    def remove(self):
        while self._undo:
            obj, name, value = self._undo.pop()
            setattr(obj, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def _keys_for_scoring(self, keys):
        return quantize_keys(keys, self.bits, self.stats_prefix)

    def _select(self, *args, **kwargs):
        shipped = self._original(*args, **kwargs)
        layer = self._stash.get("layer", 0)
        if self.layers is not None and layer not in self.layers:
            return shipped
        qk = self._stash.get("qk")
        if qk is None:
            return shipped
        q, k = qk                                             # q [tokens, query heads, D], k [cached tokens, kv heads, D]
        try:
            if q.shape[0] != 1:
                if not self.tail:
                    return shipped
                bound = self._signature.bind(*args, **kwargs); bound.apply_defaults()
                return self._prompt_tokens(shipped, q, k, int(bound.arguments["init_blocks"]),
                                           int(bound.arguments["local_blocks"]), layer)
            return self._decode_step(shipped, q, k, args, kwargs, layer)
        except Exception as e:                                # a failed hook must never pass for a fixed run
            self.stats["errors"].append(f"{type(e).__name__}: {e}")
            return shipped

    def _decode_step(self, shipped, q, k, args, kwargs, layer):
        A = _decode_arguments(self._signature, args, kwargs)
        if A is None:
            return shipped
        Hq, D = q.shape[1], q.shape[2]
        T, Hkv = k.shape[0], k.shape[1]
        group, slots, n_blocks = Hq // Hkv, shipped.shape[-1], math.ceil(T / self.block)
        scores, q_idx = _release_block_scores(self.module, A)
        if scores.shape[0] != Hkv:
            raise RuntimeError(f"the release's block scores have {scores.shape[0]} heads, expected {Hkv}")
        selection = torch.full_like(shipped, -1)
        pad = n_blocks * self.block - T
        for h in range(Hkv):
            if self.units is not None and (layer, h) not in self.units:
                selection[h] = shipped[h]
                continue
            forced, valid = _forced_and_valid(scores[h], n_blocks, q_idx)
            p = group_mass(q[0, h * group:(h + 1) * group].float(), self._keys_for_scoring(k[:, h].float()))
            if pad:
                p = torch.cat([p, p.new_zeros(pad)])
            mass = p.view(n_blocks, self.block).sum(-1)
            fill_selection(selection, h, mass.argsort(descending=True).tolist(), forced, valid, slots, shipped)
            self.stats["decode_heads"] += 1
        self.stats["decode_calls"] += 1
        return selection

    def _prompt_tokens(self, shipped, q, k, init_blocks, local_blocks, layer):
        """Rewrite the selections of the last `tail` prompt tokens in one prefill pass. shipped [kv heads, T, slots]."""
        T, Hq, D = q.shape
        Hkv = k.shape[1]
        group, block = Hq // Hkv, self.block
        K = min(self.tail, T)
        slots, n_blocks = shipped.shape[-1], math.ceil(T / block)
        pad = n_blocks * block - T
        selection = shipped.clone()
        pos = torch.arange(T - K, T, device=q.device)
        kidx = torch.arange(T, device=q.device)
        bidx = torch.arange(n_blocks, device=q.device).view(1, n_blocks)
        batched = K > 512                                     # many tokens: the selections are built in batch
        for h in range(Hkv):
            if self.units is not None and (layer, h) not in self.units:
                continue
            qt = q[T - K:, h * group:(h + 1) * group].float()                       # [K, G, D]
            kk = self._keys_for_scoring(k[:, h].float())
            rows = []
            for c0 in range(0, K, 32):
                logits = torch.einsum("kgd,td->kgt", qt[c0:c0 + 32], kk) / math.sqrt(D)
                causal = kidx.view(1, 1, T) <= pos[c0:c0 + 32].view(-1, 1, 1)
                pm = torch.softmax(logits.masked_fill(~causal, -float("inf")), dim=-1).sum(1)   # [tokens, T]
                if pad:
                    pm = torch.cat([pm, pm.new_zeros(pm.shape[0], pad)], dim=-1)
                mc = pm.view(pm.shape[0], n_blocks, block).sum(-1)                  # mass per block
                if not batched:
                    rows.append(mc)
                    continue
                pc = pos[c0:c0 + 32]
                qb = (pc // block).view(-1, 1)
                valid = bidx <= qb
                forced = ((bidx < init_blocks) | ((bidx >= qb - local_blocks) & (bidx <= qb))) & valid
                ship = torch.zeros(pc.numel(), n_blocks + 1, dtype=torch.bool, device=q.device)
                srow = shipped[h, T - K + c0:T - K + c0 + pc.numel()].long()
                ship.scatter_(1, torch.where(srow >= 0, srow, torch.full_like(srow, n_blocks)), True)
                self.stats["forced_blocks_missing"] += int((forced & ~ship[:, :n_blocks]).sum())
                rank = mc.masked_fill(~valid, -float("inf")).masked_fill(forced, float("inf"))
                top = rank.topk(min(slots, n_blocks), dim=-1)
                idx = top.indices.masked_fill(top.values == -float("inf"), n_blocks).sort(-1).values
                out = torch.full((pc.numel(), slots), -1, dtype=shipped.dtype, device=q.device)
                out[:, :idx.shape[1]] = idx.masked_fill(idx == n_blocks, -1).to(shipped.dtype)
                selection[h, T - K + c0:T - K + c0 + pc.numel()] = out
                self.stats["prefill_tokens"] += pc.numel()
            if batched:
                continue
            mass = torch.cat(rows, 0)                                               # [K, blocks]
            for i in range(K):
                p_i = int(pos[i])
                forced = forced_blocks(p_i, block, local_blocks, init_blocks)
                valid = torch.arange(n_blocks, device=q.device) <= p_i // block
                ship = set(shipped[h, T - K + i].tolist())
                self.stats["forced_blocks_missing"] += sum(1 for b in forced if b not in ship)
                selection[h, T - K + i, :] = -1
                fill_selection(selection, h, mass[i].argsort(descending=True).tolist(), forced, valid, slots, shipped,
                               row=T - K + i)
                self.stats["prefill_tokens"] += 1
        return selection
