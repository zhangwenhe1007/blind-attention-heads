"""Deployable form of the training-free low-bit key scorer (RESULTS 90.72): pure PyTorch, no custom kernel.

What the evaluated rule is (oracle_run.py --oracle_query meanq --key_bits 2 [--layers 16-23]): in a band layer, a decode
row keeps the blocks the kernel forces (init block(s), local window, the query's own block) and fills the free slots
with the blocks that carry the most softmax mass of the group-MEAN query over a 2-bit copy of the keys. Blocks are
ranked by sum_j exp(s_j), which equals ranking by the per-block logsumexp, so no softmax over the cache is needed.

How this module computes the same thing cheaply:
  * side cache per band layer: 2-bit codes packed four to a byte, [Hkv, T, D/4] uint8 (32 bytes per key for D = 128),
    and the per-channel mean and std frozen when the cache is first built (end of prefill); new decode keys are
    quantized with the frozen statistics and appended.
  * a decode step builds, per KV head, D/4 tables of 256 partial sums from w = q_mean * std * step, looks up one
    table entry per byte (32 lookups per key), sums them, and takes the logsumexp per block.
The quantizer is bit-for-bit the one in oracle_run._quantize_keys (unit-tested); the only difference from the
evaluated path is that statistics are frozen instead of recomputed every step.
"""
import inspect
import math
import sys

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

STASH = {}
STEP2 = 0.996                      # optimal uniform mid-rise step for a unit Gaussian at 2 bits (oracle_run._UNIFORM_STEP)
_LEVELS = torch.tensor([-1.5, -0.5, 0.5, 1.5])


def channel_stats(kk, prefix=None):
    """kk [..., T, D] -> (mu, sd) each [..., 1, D], exactly as oracle_run._quantize_keys computes them. prefix=N takes
    the statistics from the first N keys only (oracle_run --quant_stats_prefix): a causal quantizer."""
    ref = kk if prefix is None else kk[..., :max(int(prefix), 2), :]
    mu = ref.mean(-2, keepdim=True)
    sd = (ref - mu).pow(2).mean(-2, keepdim=True).sqrt().clamp(min=1e-8)
    return mu, sd


def encode2(kk, mu, sd):
    """2-bit codes in {0, 1, 2, 3} for kk [..., T, D] (code c stands for the level (c - 1.5) * STEP2 * sd + mu)."""
    z = torch.clamp(torch.floor((kk - mu) / (sd * STEP2)) + 0.5, -1.5, 1.5)
    return (z + 1.5).to(torch.uint8)


def pack2(codes):
    """[..., T, D] codes in 0..3 -> [..., T, D/4] bytes, channel 4b+i in bits 2i..2i+1 of byte b."""
    *lead, D = codes.shape
    if D % 4:
        raise ValueError("head dim must be a multiple of 4")
    c = codes.view(*lead, D // 4, 4).to(torch.int32)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).to(torch.uint8)


def dequant2(packed, mu, sd):
    """Inverse of encode2 + pack2 (for tests): [..., T, D/4] bytes -> [..., T, D] floats."""
    p = packed.to(torch.int32)
    codes = torch.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], dim=-1).flatten(-2)          # [..., T, D]
    return mu + (codes.float() - 1.5) * STEP2 * sd


STEPS = {2: STEP2, 4: 0.335}       # optimal uniform mid-rise steps for a unit Gaussian (oracle_run._UNIFORM_STEP)


def encode(kk, mu, sd, bits):
    """b-bit codes 0 .. 2^b - 1 (code c stands for the level (c - (2^(b-1) - 0.5)) * STEPS[b] * sd + mu)."""
    half = 2 ** (bits - 1) - 0.5
    z = torch.clamp(torch.floor((kk - mu) / (sd * STEPS[bits])) + 0.5, -half, half)
    return (z + half).to(torch.uint8)


def pack(codes, bits):
    """[..., T, D] codes -> [..., T, D * bits / 8] bytes (2 bits: four codes per byte; 4 bits: two, low nibble first)."""
    if bits == 2:
        return pack2(codes)
    *lead, D = codes.shape
    if D % 2:
        raise ValueError("head dim must be even")
    c = codes.view(*lead, D // 2, 2).to(torch.int32)
    return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)


def dequant(packed, mu, sd, bits):
    """Inverse of encode + pack: bytes -> [..., T, D] floats."""
    if bits == 2:
        return dequant2(packed, mu, sd)
    p = packed.to(torch.int32)
    codes = torch.stack([p & 15, (p >> 4) & 15], dim=-1).flatten(-2)
    return mu + (codes.float() - 7.5) * STEPS[4] * sd


def _byte_levels(device):
    v = torch.arange(256, device=device, dtype=torch.int32)
    lv = _LEVELS.to(device)
    return torch.stack([lv[((v >> s) & 3).long()] for s in (0, 2, 4, 6)], dim=-1)             # [256, 4]


def key_scores(q_mean, packed, sd, sqrt_d):
    """q_mean [H, D], packed [H, T, D/4], sd [H, 1, D] -> s [H, T] with s_j = q_mean . (dequantized k_j - mu) / sqrt_d.
    The channel means only add a per-head constant, which no ranking or softmax over keys can see, so it is dropped."""
    H, T, nb = packed.shape
    w = (q_mean * sd.squeeze(-2) * STEP2 / sqrt_d).view(H, nb, 4)                             # [H, D/4, 4]
    tables = torch.einsum("hbi,vi->hbv", w, _byte_levels(packed.device).to(w.dtype))          # [H, D/4, 256]
    got = torch.gather(tables, 2, packed.permute(0, 2, 1).long())                             # [H, D/4, T]
    return got.sum(1)


def cache_scores(q, cache, sqrt_d):
    """Reference scores for any bit width. q [H, G, D] (G = 1 for a mean query) -> [H, G, T]:
    q . (dequantized key - channel mean) / sqrt_d."""
    kd = dequant(cache.packed(), cache.mu, cache.sd, cache.bits) - cache.mu                 # [H, T, D]
    return torch.bmm(q, kd.transpose(1, 2)) / sqrt_d


def head_scores(q_heads, packed, sd, sqrt_d):
    """Per QUERY head scores. q_heads [H, G, D], packed [H, T, D/4], sd [H, 1, D] -> s [H, G, T] with
    s_gj = q_g . (dequantized k_j - mu) / sqrt_d. Unlike the mean-query path the per-head constants q_g . mu matter
    for nothing either: each head's softmax is normalized on its own."""
    H, G, D = q_heads.shape
    out = [key_scores(q_heads[:, g], packed, sd, sqrt_d) for g in range(G)]
    return torch.stack(out, dim=1)


def group_mass_from_block_lse(lse):
    """lse [H, G, n_blocks]: per-head per-block logsumexp. Returns log of the GROUP mass per block [H, n_blocks]:
    log sum_g exp(lse_gb - total_g), the exact criterion (each head's softmax mass of the block, summed over the group)."""
    total = torch.logsumexp(lse, dim=-1, keepdim=True)
    return torch.logsumexp(lse - total, dim=1)


def block_lse(s, n_blocks, block):
    """s [H, T] -> per-block logsumexp [H, n_blocks]; positions past T are padding."""
    H, T = s.shape
    pad = n_blocks * block - T
    if pad:
        s = torch.cat([s, s.new_full((H, pad), -float("inf"))], dim=-1)
    return torch.logsumexp(s.view(H, n_blocks, block), dim=-1)


def forced_mask(n_blocks, q_block, local_blocks, init_blocks, device):
    """Blocks the kernel always reads for a decode query in block q_block (34 of them at 32K for 1 / 32 / own)."""
    b = torch.arange(n_blocks, device=device)
    return (b < init_blocks) | ((b >= q_block - local_blocks) & (b <= q_block))


def select_rows(scores, q_block, width, local_blocks, init_blocks):
    """scores [H, n_blocks] -> int32 rows [H, 1, width], index-sorted, padded with -1: every forced block plus the
    best-scoring free blocks up to the row width. Blocks after the query's block are never selected."""
    H, n_blocks = scores.shape
    forced = forced_mask(n_blocks, q_block, local_blocks, init_blocks, scores.device)
    valid = torch.arange(n_blocks, device=scores.device) <= q_block
    rank = scores.masked_fill(~valid, -float("inf")).masked_fill(forced & valid, float("inf"))
    k = min(width, int(valid.sum()))
    idx = rank.topk(k, dim=-1).indices.sort(-1).values.to(torch.int32)
    out = torch.full((H, 1, width), -1, dtype=torch.int32, device=scores.device)
    out[:, 0, :k] = idx
    return out


class SideCache:
    """Packed low-bit keys of one layer, [Hkv, cap, D * bits / 8], with statistics frozen at build time."""

    def __init__(self, keys, extra=256, bits=2, stats_prefix=None):
        # keys: [T, Hkv, D] (the layout of the harness's stash)
        if bits not in STEPS:
            raise ValueError("bits must be 2 or 4")
        self.bits = bits
        kk = keys.permute(1, 0, 2).float()                                                  # [Hkv, T, D]
        self.mu, self.sd = channel_stats(kk, stats_prefix)
        self.T = kk.shape[1]
        packed = pack(encode(kk, self.mu, self.sd, bits), bits)
        self.buf = torch.zeros(packed.shape[0], self.T + extra, packed.shape[2], dtype=torch.uint8, device=keys.device)
        self.buf[:, :self.T] = packed

    def append(self, new_keys):
        """new_keys: [t, Hkv, D], the keys of the tokens decoded since the last call."""
        t = new_keys.shape[0]
        if self.T + t > self.buf.shape[1]:
            grow = torch.zeros(self.buf.shape[0], max(t, 256), self.buf.shape[2], dtype=torch.uint8, device=self.buf.device)
            self.buf = torch.cat([self.buf, grow], dim=1)
        self.buf[:, self.T:self.T + t] = pack(encode(new_keys.permute(1, 0, 2).float(), self.mu, self.sd, self.bits), self.bits)
        self.T += t

    def packed(self):
        return self.buf[:, :self.T]


def decode_rows(cache, q, width, block, local_blocks, init_blocks):
    """One decode step of one band layer. q [1, Hq, D] the decode queries; returns rows [Hkv, 1, width] int32."""
    Hq, D = q.shape[1], q.shape[2]
    Hkv = cache.buf.shape[0]
    q_mean = q[0].float().view(Hkv, Hq // Hkv, D).mean(1)                                     # [Hkv, D]
    s = key_scores(q_mean, cache.packed(), cache.sd, math.sqrt(D))
    n_blocks = math.ceil(cache.T / block)
    return select_rows(block_lse(s, n_blocks, block), (cache.T - 1) // block, width, local_blocks, init_blocks)


# ---------------------------------------------------------------------------------------------------------------------
# GPU path: one fused kernel per layer reads the packed codes and writes the per-block logsumexp.
# ---------------------------------------------------------------------------------------------------------------------
if HAVE_TRITON:
    @triton.jit
    def _packed_block_lse_kernel(CODES, W, OUT, T, s_ch, s_ct, s_wh, s_oh, BLOCK: tl.constexpr, NB: tl.constexpr):
        h = tl.program_id(0)
        b = tl.program_id(1)
        rows = b * BLOCK + tl.arange(0, BLOCK)
        cols = tl.arange(0, NB)
        valid = rows < T
        c = tl.load(CODES + h * s_ch + rows[:, None] * s_ct + cols[None, :], mask=valid[:, None], other=0).to(tl.int32)
        w0 = tl.load(W + h * s_wh + cols * 4)
        w1 = tl.load(W + h * s_wh + cols * 4 + 1)
        w2 = tl.load(W + h * s_wh + cols * 4 + 2)
        w3 = tl.load(W + h * s_wh + cols * 4 + 3)
        # code c stands for the level c - 1.5; the -1.5 * sum(w) term is one constant per head and is dropped
        s = tl.sum((c & 3).to(tl.float32) * w0[None, :] + ((c >> 2) & 3).to(tl.float32) * w1[None, :]
                   + ((c >> 4) & 3).to(tl.float32) * w2[None, :] + ((c >> 6) & 3).to(tl.float32) * w3[None, :], axis=1)
        s = tl.where(valid, s, float("-inf"))
        mx = tl.max(s, axis=0)
        tl.store(OUT + h * s_oh + b, mx + tl.log(tl.sum(tl.exp(s - mx), axis=0)))


if HAVE_TRITON:
    @triton.jit
    def _packed4_block_lse_kernel(CODES, W, OUT, T, s_ch, s_ct, s_wh, s_oh, BLOCK: tl.constexpr, NB: tl.constexpr):
        # 4-bit codes, two per byte (low nibble = even channel); code c stands for the level c - 7.5, and the
        # -7.5 * sum(w) term is one constant per head, dropped as in the 2-bit kernel
        h = tl.program_id(0)
        b = tl.program_id(1)
        rows = b * BLOCK + tl.arange(0, BLOCK)
        cols = tl.arange(0, NB)
        valid = rows < T
        c = tl.load(CODES + h * s_ch + rows[:, None] * s_ct + cols[None, :], mask=valid[:, None], other=0).to(tl.int32)
        w0 = tl.load(W + h * s_wh + cols * 2)
        w1 = tl.load(W + h * s_wh + cols * 2 + 1)
        s = tl.sum((c & 15).to(tl.float32) * w0[None, :] + ((c >> 4) & 15).to(tl.float32) * w1[None, :], axis=1)
        s = tl.where(valid, s, float("-inf"))
        mx = tl.max(s, axis=0)
        tl.store(OUT + h * s_oh + b, mx + tl.log(tl.sum(tl.exp(s - mx), axis=0)))


if HAVE_TRITON:
    @triton.jit
    def _masked_block_lse_kernel(S, POS, OUT, n_rows, t1, nb1, s_sn, s_on, BLOCK: tl.constexpr, NBLK: tl.constexpr):
        # S [N, t1] float16 scores of N = G * n_rows query rows (row n belongs to prompt position POS[n % n_rows]);
        # one program = one row and NBLK consecutive key blocks: causal mask and per-block logsumexp in a single pass
        n = tl.program_id(0)
        j = tl.program_id(1)
        p = tl.load(POS + (n % n_rows))
        b = j * NBLK + tl.arange(0, NBLK)
        cols = b[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
        ok = (cols <= p) & (cols < t1)
        s = tl.load(S + n * s_sn + cols, mask=ok, other=float("-inf")).to(tl.float32)
        mx = tl.max(s, axis=1)
        mxs = tl.where(mx == float("-inf"), 0.0, mx)              # a fully masked block: exp(-inf) sums to 0, log gives -inf
        lse = mxs + tl.log(tl.sum(tl.exp(s - mxs[:, None]), axis=1))
        tl.store(OUT + n * s_on + b, lse, mask=b < nb1)


def triton_masked_block_lse(sc, pos, block, nblk=8):
    """sc [N, t1] float16, contiguous, t1 a multiple of `block`, N = G * len(pos) with the query head as the slow index;
    pos [c] int64 prompt positions. Returns float32 [N, t1 // block]: logsumexp of each block over the keys <= pos."""
    N, t1 = sc.shape
    nb1 = t1 // block
    out = torch.empty(N, nb1, device=sc.device, dtype=torch.float32)
    _masked_block_lse_kernel[(N, triton.cdiv(nb1, nblk))](sc, pos, out, pos.numel(), t1, nb1, sc.stride(0), out.stride(0),
                                                           BLOCK=block, NBLK=nblk)
    return out


def _launch(bits, grid, *args, **kw):
    (_packed_block_lse_kernel if bits == 2 else _packed4_block_lse_kernel)[grid](*args, **kw)


def triton_group_scores(q_heads, buf, T, sd, sqrt_d, n_blocks, block, bits=2):
    """Per-head criterion on the packed keys with the same kernel: the G query heads of each KV head are laid out as
    H*G pseudo heads that share the KV head's codes (strides select the shared buffer). Returns [H, n_blocks], the log
    group mass per block (group_mass_from_block_lse)."""
    H, G, D = q_heads.shape
    nb = buf.shape[2]
    w = (q_heads * sd * STEPS[bits] / sqrt_d).float().reshape(H * G, D).contiguous() # sd [H, 1, D] broadcasts over G
    out = torch.empty(H * G, n_blocks, device=buf.device, dtype=torch.float32)
    for h in range(H):                                                             # one launch per KV head, G programs wide
        _launch(bits, (G, n_blocks), buf[h:h + 1], w[h * G:(h + 1) * G], out[h * G:(h + 1) * G], T,
                0, buf.stride(1), w.stride(0), out.stride(0), BLOCK=block, NB=nb)
    return group_mass_from_block_lse(out.view(H, G, n_blocks))


def triton_block_lse(q_mean, buf, T, sd, sqrt_d, n_blocks, block, bits=2):
    """q_mean [H, D], buf [H, cap, D/4] uint8 (cap >= T), sd [H, 1, D] -> [H, n_blocks] fp32, equal to
    block_lse(key_scores(...)) up to one additive constant per head."""
    H, _, nb = buf.shape
    w = (q_mean * sd.squeeze(-2) * STEPS[bits] / sqrt_d).float().contiguous()                # [H, D]
    out = torch.empty(H, n_blocks, device=buf.device, dtype=torch.float32)
    _launch(bits, (H, n_blocks), buf, w, out, T, buf.stride(0), buf.stride(1), w.stride(0), out.stride(0),
            BLOCK=block, NB=nb)
    return out


def _fingerprint(k, T0):
    """A few key entries spread over the prompt (head 0): two prompts of equal length do not share them."""
    pos = torch.tensor([T0 // 4, T0 // 2, (3 * T0) // 4, T0 - 1], device=k.device)
    return k[pos, 0, :8].float().clone()


def prefill_tail_rows(cache, q_tail, first_pos, width, block, local_blocks, init_blocks, criterion="meanq"):
    """Rows for the LAST K prefill positions (the question) from the packed side cache, in one pass.
    q_tail [K, Hq_sel, D] are the tail queries of the heads that share the cache's KV heads (Hq_sel = Hkv_sel * grp),
    first_pos is the position of the first tail query. Every query scores the keys up to its own position with its
    group-mean query against the dequantized 2-bit keys (the same scores as the decode scan) and takes the per-block
    logsumexp; rows are built by select_rows. Returns int32 [Hkv_sel, K, width]."""
    K, Hq, D = q_tail.shape
    H = cache.buf.shape[0]
    T = cache.T
    n_blocks = math.ceil(T / block)
    kd = dequant(cache.packed(), cache.mu, cache.sd, cache.bits) - cache.mu            # [H, T, D]; the mean is a constant per query
    qg = q_tail.float().view(K, H, Hq // H, D).permute(1, 2, 0, 3)                     # [H, G, K, D]
    if criterion == "meanq":
        qg = qg.mean(1, keepdim=True)
    pos = first_pos + torch.arange(K, device=kd.device)
    future = torch.arange(T, device=kd.device).view(1, 1, T) > pos.view(1, K, 1)
    pad = n_blocks * block - T
    per_head = []
    for g in range(qg.shape[1]):                                                       # one query head at a time: [H, K, T]
        s = (torch.bmm(qg[:, g], kd.transpose(1, 2)) / math.sqrt(D)).masked_fill(future, -float("inf"))
        if pad:
            s = torch.cat([s, s.new_full((H, K, pad), -float("inf"))], dim=-1)
        per_head.append(torch.logsumexp(s.view(H, K, n_blocks, block), dim=-1))        # [H, K, n_blocks]
    if criterion == "meanq":
        lse = per_head[0]
    else:                                                                              # log of the group mass per block
        st = torch.stack(per_head, dim=2)                                              # [H, K, G, n_blocks]
        lse = torch.logsumexp(st - torch.logsumexp(st, dim=-1, keepdim=True), dim=2)
    out = torch.full((H, K, width), -1, dtype=torch.int32, device=s.device)
    for i in range(K):
        out[:, i] = select_rows(lse[:, i], int(pos[i]) // block, width, local_blocks, init_blocks)[:, 0]
    return out


def select_rows_batched(scores, q_blocks, width, local_blocks, init_blocks):
    """select_rows for many query rows at once. scores [H, K, n_blocks], q_blocks [K] (the block of each query) ->
    int32 [H, K, width], every row index-sorted and padded with -1, equal to select_rows row by row."""
    H, K, n_blocks = scores.shape
    b = torch.arange(n_blocks, device=scores.device).view(1, n_blocks)
    qb = q_blocks.view(K, 1)
    valid = b <= qb                                                                     # [K, n_blocks]
    forced = ((b < init_blocks) | ((b >= qb - local_blocks) & (b <= qb))) & valid
    rank = scores.masked_fill(~valid.unsqueeze(0), -float("inf")).masked_fill(forced.unsqueeze(0), float("inf"))
    k = min(width, n_blocks)
    top = rank.topk(k, dim=-1)
    idx = top.indices.masked_fill(top.values == -float("inf"), n_blocks)                # a block after the query: drop it
    idx = idx.sort(-1).values
    out = torch.full((H, K, width), -1, dtype=torch.int32, device=scores.device)
    out[:, :, :k] = idx.masked_fill(idx == n_blocks, -1).to(torch.int32)
    return out


def prefill_rows_chunked(cache, q_rows, first_pos, width, block, local_blocks, init_blocks, criterion="heads", chunk=256,
                         half=False, fused=False):
    """prefill_tail_rows without a Python loop over rows and without materializing every row's scores at once, so that
    it can rewrite EVERY prefill row of the listed heads (first_pos = 0, K = T). q_rows [K, Hq_sel, D]. Rows are scored
    `chunk` at a time against the keys up to the chunk's last position. Returns int32 [Hkv_sel, K, width].
    half=True scores in float16 (the products and the per-block logsumexp; the group mass is combined in float32). A
    float16 score near 32 is exact to 0.03 nats, a third of a percent of the 4-bit quantization noise itself; the keys are
    padded to a whole number of blocks once, so no chunk concatenates padding."""
    K, Hq, D = q_rows.shape
    H = cache.buf.shape[0]
    T = cache.T
    n_blocks = math.ceil(T / block)
    kd = dequant(cache.packed(), cache.mu, cache.sd, cache.bits) - cache.mu            # [H, T, D]
    qg = q_rows.float().view(K, H, Hq // H, D).permute(1, 2, 0, 3)                     # [H, G, K, D]
    if criterion == "meanq":
        qg = qg.mean(1, keepdim=True)
    G = qg.shape[1]
    qg = qg / math.sqrt(D)
    if n_blocks * block > T:                                                            # whole blocks: padded keys are masked
        kd = torch.cat([kd, kd.new_zeros(H, n_blocks * block - T, D)], dim=1)
    if half:
        kd, qg = kd.half(), qg.half()
    kdt = kd.transpose(1, 2)
    out = torch.full((H, K, width), -1, dtype=torch.int32, device=kd.device)
    for c0 in range(0, K, chunk):
        c1 = min(c0 + chunk, K)
        pos = first_pos + torch.arange(c0, c1, device=kd.device)
        nb1 = int(pos[-1]) // block + 1                                                 # blocks this chunk can see
        t1 = nb1 * block
        future = torch.arange(t1, device=kd.device).view(1, 1, t1) > pos.view(1, -1, 1)
        if half:
            # one float16 matrix product per KV head for all G query heads of the chunk, then a masked per-block
            # logsumexp computed in place (no full-size temporaries): amax, subtract, exp, sum, log
            c = c1 - c0
            fut4 = future.view(1, c, nb1, block)
            per_kv = []
            for h in range(H):
                if fused:                                                   # mask + block logsumexp in one Triton pass
                    sc = torch.mm(qg[h, :, c0:c1].reshape(G * c, D), kdt[h, :, :t1])
                    per_kv.append(triton_masked_block_lse(sc, pos, block).view(G, c, nb1))
                    continue
                v = torch.mm(qg[h, :, c0:c1].reshape(G * c, D), kdt[h, :, :t1]).view(G, c, nb1, block)
                v.masked_fill_(fut4, -float("inf"))
                mx = v.amax(-1, keepdim=True).clamp_(min=-6e4)             # a fully masked block stays finite here ...
                per_kv.append(v.sub_(mx).exp_().sum(-1).log_().add_(mx.squeeze(-1)).float())   # ... and ends at -inf
            st = torch.stack(per_kv, 0).permute(0, 2, 1, 3)                 # [H, c, G, nb1]
            lse = st[:, :, 0] if G == 1 else torch.logsumexp(st - torch.logsumexp(st, dim=-1, keepdim=True), dim=2)
            if nb1 < n_blocks:
                lse = torch.cat([lse, lse.new_full((H, c, n_blocks - nb1), -float("inf"))], dim=-1)
            out[:, c0:c1] = select_rows_batched(lse, pos // block, width, local_blocks, init_blocks)
            continue
        per_head = []
        for g in range(G):
            sc = torch.bmm(qg[:, g, c0:c1], kdt[:, :, :t1]).masked_fill_(future, -float("inf"))
            per_head.append(torch.logsumexp(sc.view(H, c1 - c0, nb1, block), dim=-1).float())   # [H, c, nb1]
        if G == 1:
            lse = per_head[0]
        else:
            st = torch.stack(per_head, dim=2)                                           # [H, c, G, nb1]
            lse = torch.logsumexp(st - torch.logsumexp(st, dim=-1, keepdim=True), dim=2)
        if nb1 < n_blocks:
            lse = torch.cat([lse, lse.new_full((H, c1 - c0, n_blocks - nb1), -float("inf"))], dim=-1)
        out[:, c0:c1] = select_rows_batched(lse, pos // block, width, local_blocks, init_blocks)
    return out


def install(model, cfg, layers=None, mode="torch", log=None, kv_heads=None, prefill_tail=0, criterion="meanq", bits=2,
            stats_prefix=None, half_prefill=False, fused_prefill=False):
    """Patch the decode-time selector. In the band layers (layers=(lo, hi) inclusive or a set, None = every layer) a decode
    step never calls the shipped stage-1 selector: its row is built from the packed side cache. Prefill and the other
    layers are untouched. Returns a state dict (n_calls, n_passthrough, errors, stale).
    `log` (a list) receives (tk_len, layer, [per-head sorted picks]) for the equivalence check.
    kv_heads: a set of (layer, KV head) pairs (replaces `layers`). Only those heads get the scorer's row and only their
    keys are packed; the layer's other heads keep the shipped row, so the shipped stage-1 still runs in that layer."""
    if mode == "triton" and not HAVE_TRITON:
        raise SystemExit("triton not importable")
    block = cfg.sparse_config["block_size"]
    mod = sys.modules[type(model).__module__]
    orig_ca = mod.compressed_attention
    sig = inspect.signature(orig_ca)
    if kv_heads is not None:
        if layers is not None:
            raise ValueError("give layers or kv_heads, not both")
        layers = frozenset(l for l, _ in kv_heads)
    heads_of = {l: sorted(h for ll, h in kv_heads if ll == l) for l in layers} if kv_heads is not None else None
    if criterion not in ("meanq", "heads"):
        raise ValueError("criterion must be meanq or heads")
    if bits not in STEPS:
        raise ValueError("bits must be 2 or 4")
    state = {"n_calls": 0, "n_passthrough": 0, "errors": [], "mode": mode, "layers": layers, "kv_heads": kv_heads,
             "criterion": criterion, "bits": bits, "stats_prefix": stats_prefix, "half_prefill": half_prefill, "fused_prefill": fused_prefill,
             "log_prefill_rows": None,
             "caches": {}, "stale": None, "orig_ca": orig_ca, "orig_sf": []}

    def ca(*a, **kw):
        qk = STASH.get("qk")
        li = STASH.get("li", 0)
        in_band = layers is None or (li in layers if isinstance(layers, (set, frozenset)) else layers[0] <= li <= layers[1])
        if qk is None or not in_band:
            state["n_passthrough"] += 1
            return orig_ca(*a, **kw)
        q, k, _v = qk
        if q.shape[0] != 1:                                   # prefill: shipped selector, and the side cache is stale
            state["caches"].pop(li, None)
            state["n_passthrough"] += 1
            rows_p = orig_ca(*a, **kw)
            if not prefill_tail or isinstance(rows_p, tuple):
                return rows_p
            try:                                              # the question rows come from the scorer, same pass
                B = sig.bind(*a, **kw); B.apply_defaults()
                A = B.arguments
                T = k.shape[0]
                K = min(prefill_tail, T)
                hs = torch.tensor(heads_of[li], device=k.device) if heads_of is not None else torch.arange(k.shape[1], device=k.device)
                cache = SideCache(k[:, hs], bits=bits, stats_prefix=stats_prefix)
                cache.fp = _fingerprint(k[:, hs], T)
                cache.T0 = T
                state["caches"][li] = cache                   # reused by the first decode step (one key appended)
                grp = q.shape[1] // k.shape[1]
                qsel = q[T - K:].view(K, k.shape[1], grp, q.shape[2])[:, hs].reshape(K, -1, q.shape[2])
                if K > 512:                                   # many rows: the batched, chunked path
                    tail = prefill_rows_chunked(cache, qsel, T - K, rows_p.shape[-1], block, int(A["local_blocks"]),
                                                int(A["init_blocks"]), criterion=criterion, chunk=(512 if half_prefill else 256),
                                                half=half_prefill, fused=(fused_prefill and half_prefill and HAVE_TRITON))
                else:
                    tail = prefill_tail_rows(cache, qsel, T - K, rows_p.shape[-1], block, int(A["local_blocks"]),
                                             int(A["init_blocks"]), criterion=criterion)
                rows_p = rows_p.clone()
                rows_p[hs, T - K:] = tail.to(rows_p.dtype)
                if state.get("log_prefill_rows") is not None:     # bench_qk's gate compares these with the evaluated path's
                    for j, h in enumerate(hs.tolist()):
                        state["log_prefill_rows"][(int(li), h)] = tail[j].to("cpu", torch.int32)
                state["n_tail_rows"] = state.get("n_tail_rows", 0) + int(tail.shape[0] * tail.shape[1])
            except Exception as e:
                state["errors"].append(f"prefill_tail {type(e).__name__}: {e}")
            return rows_p
        shipped = None
        if heads_of is not None:                               # the unlisted heads of this layer need their shipped rows
            shipped = orig_ca(*a, **kw)
            if isinstance(shipped, tuple):
                raise RuntimeError("kv_heads is implemented for tensor-returning selectors only")
            hsel = torch.tensor(heads_of[li], device=k.device)
            k = k[:, hsel]                                     # pack and scan the listed heads only
        try:
            B = sig.bind(*a, **kw); B.apply_defaults()
            A = B.arguments
            tk_len = k.shape[0]
            cache = state["caches"].get(li)
            if cache is None or tk_len != cache.T + 1:
                cache = SideCache(k, bits=bits, stats_prefix=stats_prefix)
                cache.fp = _fingerprint(k, tk_len)
                cache.T0 = tk_len
                state["caches"][li] = cache
            else:
                bad = (_fingerprint(k, cache.T0) != cache.fp).any()      # stays on the GPU: no sync per step
                state["stale"] = bad if state["stale"] is None else (state["stale"] | bad)
                cache.append(k[cache.T:tk_len])
            n_blocks = math.ceil(tk_len / block)
            width = min(int(A["topk"]), n_blocks)
            Hq, D = q.shape[1], q.shape[2]
            Hkv_all = qk[1].shape[1]
            q_grp = q[0].float().view(Hkv_all, Hq // Hkv_all, D)
            if heads_of is not None:
                q_grp = q_grp[hsel]
            if criterion == "heads":                          # each query head's own softmax over the low-bit keys, summed
                if mode == "triton":
                    scores = triton_group_scores(q_grp, cache.buf, cache.T, cache.sd, math.sqrt(D), n_blocks, block, bits)
                else:
                    sh = cache_scores(q_grp, cache, math.sqrt(D))                          # [H, G, T]
                    lse = torch.stack([block_lse(sh[:, g], n_blocks, block) for g in range(sh.shape[1])], dim=1)
                    scores = group_mass_from_block_lse(lse)
            elif mode == "triton":
                scores = triton_block_lse(q_grp.mean(1), cache.buf, cache.T, cache.sd, math.sqrt(D), n_blocks, block, bits)
            elif bits == 2:
                scores = block_lse(key_scores(q_grp.mean(1), cache.packed(), cache.sd, math.sqrt(D)), n_blocks, block)
            else:
                scores = block_lse(cache_scores(q_grp.mean(1, keepdim=True), cache, math.sqrt(D))[:, 0], n_blocks, block)
            rows = select_rows(scores, (tk_len - 1) // block, width, int(A["local_blocks"]), int(A["init_blocks"]))
            if heads_of is not None:
                if shipped.shape[-1] != rows.shape[-1]:
                    raise RuntimeError(f"row widths differ: shipped {shipped.shape[-1]} scorer {rows.shape[-1]}")
                out_rows = shipped.clone()
                out_rows[hsel] = rows.to(shipped.dtype)
                rows = out_rows
            if log is not None:
                log.append((tk_len, li, [sorted(x for x in r if x >= 0) for r in rows[:, 0].tolist()]))
            state["n_calls"] += 1
            return rows
        except Exception as e:                                # never silently: callers must abort on state["errors"]
            state["errors"].append(f"{type(e).__name__}: {e}")
            return shipped if shipped is not None else orig_ca(*a, **kw)
    mod.compressed_attention = ca

    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig_sf = attn.sparse_forward
        state["orig_sf"].append((attn, orig_sf))

        def make(orig_sf, li):
            def sf(q, k, v, *a, **kw):
                STASH["qk"] = (q, k, v); STASH["li"] = li
                return orig_sf(q, k, v, *a, **kw)
            return sf
        attn.sparse_forward = make(orig_sf, li)
    return state


def reset(state):
    """Call before each new prompt (the prefill hook and the fingerprint are the backstops)."""
    state["caches"].clear()


def stale_uses(state):
    """True if any decode step appended to a side cache that belonged to a different prompt (one sync; call at the
    end of a sample or a run and abort on True)."""
    return bool(state["stale"]) if state["stale"] is not None else False


def uninstall(model, state):
    mod = sys.modules[type(model).__module__]
    mod.compressed_attention = state["orig_ca"]
    for attn, orig_sf in state["orig_sf"]:
        attn.sparse_forward = orig_sf
    state["caches"].clear()
    STASH.clear()
