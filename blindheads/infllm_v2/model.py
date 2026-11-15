"""Load an InfLLM-V2 checkpoint, generate with it and score RULER prompts.

The release selects blocks whenever its configuration carries `sparse_config`, and attends to every key when it does
not. Both settings load identical weights, so the dense model of the paper is the same checkpoint without that entry.

InfLLM-V2-Long-Sparse-Base ships with `sparse_config` in its configuration. MiniCPM4-8B and MiniCPM4.1-8B ship without
it, so they load dense by default, and their model cards give the entry that switches the selector on. `load` adds that
entry when the checkpoint has none.
"""
import torch

from .. import ruler

# the selector setting of the model cards (and of InfLLM-V2-Long-Sparse-Base's own configuration)
SPARSE_CONFIG = {"kernel_size": 32, "kernel_stride": 16, "init_blocks": 1, "block_size": 64, "window_size": 2048,
                 "topk": 64, "use_nope": False, "dense_len": 8192}


def load(model_dir, dense=False, sparse_config=None, dtype=torch.bfloat16, device="cuda"):
    """(model, tokenizer, config) of the checkpoint in `model_dir` (a local folder or a hub id).

    dense=False: the sparse selector, with the checkpoint's own `sparse_config`, or SPARSE_CONFIG if it ships without one,
    or the `sparse_config` given here.  dense=True: full attention in every head, on the same weights.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if not hasattr(cfg, "sparse_config"):
        raise ValueError(f"the configuration of {model_dir} has no sparse_config field: not an InfLLM-V2 checkpoint")
    if dense:
        cfg.sparse_config = None
    elif sparse_config is not None:
        cfg.sparse_config = dict(sparse_config)
    elif cfg.sparse_config is None:
        cfg.sparse_config = dict(SPARSE_CONFIG)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, trust_remote_code=True, torch_dtype=dtype,
                                                 attn_implementation="flash_attention_2").to(device).eval()
    # without sparse_config the release builds its plain FlashAttention class, which has no selector to hook
    if not dense and not hasattr(model.model.layers[0].self_attn, "sparse_forward"):
        raise RuntimeError("the attention class has no sparse_forward: this release is not supported by the hooks")
    return model, tok, cfg


@torch.inference_mode()
def generate(model, tok, text, max_new_tokens):
    """Greedy continuation of `text`, decoded without special tokens."""
    enc = tok(text, return_tensors="pt")
    ids = enc.input_ids.to(model.device)
    out = model.generate(input_ids=ids, attention_mask=enc.attention_mask.to(model.device), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def evaluate(model, tok, rows, max_new_tokens=ruler.TOKENS_TO_GENERATE["niah"], metric=ruler.string_match_all, log=None):
    """Run RULER rows and score them. Returns {"accuracy": percent, "scores": [per prompt], "predictions": [...]}."""
    preds, refs = [], []
    for i, r in enumerate(rows):
        preds.append(generate(model, tok, ruler.prompt_text(r), max_new_tokens))
        refs.append(r["outputs"])
        if log is not None and (i % 10 == 0 or i == len(rows) - 1):
            log(f"  [{i + 1}/{len(rows)}] accuracy so far {ruler.accuracy(preds, refs, metric):.1f}")
    return {"accuracy": ruler.accuracy(preds, refs, metric), "scores": [metric(p, r) for p, r in zip(preds, refs)],
            "predictions": preds}
