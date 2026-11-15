"""Refuse partial training checkpoints at load time.

The indexer trainers write periodic checkpoints to the same indexer.pt path as the final save.
Periodic saves carry a top-level "progress" field that the final save omits. The gate-projection
trainer keeps "progress" under "meta" and marks its final save with sample "final". A preempted
and requeued job that treats an existing checkpoint as finished would otherwise evaluate a
half-trained module without any error.
"""
import torch


def check_final(ck, path):
    """Return ck unchanged, or raise SystemExit if it is a periodic (partial) checkpoint."""
    if not isinstance(ck, dict):
        return ck
    if "progress" in ck:
        raise SystemExit(f"{path} is a partial training checkpoint "
                         f"(progress={ck['progress']}) -- ABORT")
    meta = ck.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("progress"), dict) \
            and meta["progress"].get("sample") != "final":
        raise SystemExit(f"{path} is a partial training checkpoint "
                         f"(meta.progress={meta['progress']}) -- ABORT")
    return ck


def load_final(path, map_location="cpu"):
    return check_final(torch.load(path, map_location=map_location), path)
