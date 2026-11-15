"""fasa-org MiniCPM-4-8B-* compatibility (the DashAttention paper's retrained
8B arms, arXiv 2605.18753).

The InfLLMv2 arm is MiniCPM4-8B re-expressed as SparseLlamaForCausalLM with the
InfLLMv2 machinery in its own modeling file. Differences from our protagonist
that the harness must bridge:
  - no config.sparse_config: sparse params are hardcoded class attrs
    (kernel 32/16, block 64, window 1024, topk 64, dense_len 0);
  - every attn_implementation maps to the SPARSE class, so there is no dense
    arm by config; their compressed_attention clamps topk to the block count,
    so a huge per-layer topk IS the dense-equivalent arm;
  - the selection stash point is `_sparse_attention_forward` with [b, s, H, D]
    layout instead of `sparse_forward` (varlen [t, H, D]);
  - per-layer `self.topk` is read at call time (config topk is ignored).
Tokenizer is byte-identical to InfLLM-V2-Long-Sparse-Base (verified sha256), so
the protagonist's pinned RULER draws are reused as-is.
"""

FASA_SPARSE_CONFIG = {"kernel_size": 32, "kernel_stride": 16, "init_blocks": 1,
                      "block_size": 64, "window_size": 1024, "topk": 64,
                      "use_nope": False, "dense_len": 0}


def is_fasa(cfg):
    return "SparseLlama" in (getattr(cfg, "architectures", None) or [""])[0]


def ensure_sparse_config(cfg):
    """Synthesize sparse_config from the modeling file's hardcoded defaults.
    Also flip use_cache back on: the release config ships use_cache=False (a
    training setting), and the decode-side selection interception requires
    cached generation (tq=1 steps)."""
    if is_fasa(cfg):
        if getattr(cfg, "sparse_config", None) is None:
            if hasattr(cfg, "sigma"):
                # Dash-8B variant: block 32 (the release's modeling hardcodes it;
                # a _64 variant file exists but is not the auto_map target),
                # kernels ship inside the checkpoint, always-sparse
                cfg.sparse_config = dict(FASA_SPARSE_CONFIG, block_size=32)
            else:
                cfg.sparse_config = dict(FASA_SPARSE_CONFIG)
        cfg.use_cache = True
        _shim_kernels()
        if hasattr(cfg, "sigma"):
            _shim_author_logpath()
    return cfg


def _shim_author_logpath():
    """The Dash-8B modeling file opens logging.FileHandler on a hardcoded path
    in the original author's home directory AT IMPORT TIME (their 'average
    activated' debug log), which FileNotFoundErrors everywhere else. Redirect
    exactly that path prefix to devnull; all other FileHandler use untouched."""
    import logging as _lg
    import os
    if getattr(_lg, "_fasa_logpath_shimmed", False):
        return
    _orig = _lg.FileHandler

    class _H(_orig):
        def __init__(self, filename, *a, **kw):
            if isinstance(filename, str) and filename.startswith("/home/test/"):
                filename = os.devnull
            super().__init__(filename, *a, **kw)

    _lg.FileHandler = _H
    _lg._fasa_logpath_shimmed = True


def _shim_kernels():
    """The fasa modeling file targets an older infllm_v2 API: its stage-1 call
    omits cu_seqlens_v (its v IS compressed k, sharing cu_seqlens_k). The
    installed package requires it. Wrap the entry point to default it in;
    must run BEFORE the modeling module is imported so its `from infllm_v2
    import ...` binds the wrapper. Callers that already pass cu_seqlens_v
    (our protagonist) pass through unchanged."""
    import infllm_v2 as iv
    if getattr(iv, "_fasa_shimmed", False):
        return
    _orig = iv.infllmv2_attn_stage1

    def stage1(*a, **kw):
        if "cu_seqlens_v" not in kw and len(a) < 6:
            kw["cu_seqlens_v"] = kw.get("cu_seqlens_k")
        return _orig(*a, **kw)

    iv.infllmv2_attn_stage1 = stage1
    iv._fasa_shimmed = True


def apply_topk(model, topk):
    """fasa reads per-layer self.topk at call time; config topk is ignored."""
    for l in model.model.layers:
        l.self_attn.topk = int(topk)
