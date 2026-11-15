"""Which keys compete for the FREE slots of an InfLLM-V2 decode row.

The kernel always reads the init block(s), the local window and the query's own block (34 of 96 blocks at 32K for
init_blocks=1, window 2048, block 64, measured as forced_mean 34.0 in every corrected cell). A learned indexer only
ranks the remaining free slots. Distilling it over ALL keys spends most of the cross-entropy on the local window,
whose mass is high and whose blocks are read anyway: at the default union a third of the indexer's top picks were
forced blocks (RESULTS 90.38). free_key_mask marks the keys outside the forced region so that the distillation
target and the student softmax can be restricted to them (indexer_train.py --target_region free).
"""
import torch


def free_key_mask(pos, T, block=64, local_blocks=32, init_blocks=1):
    """pos [P] query token positions, T keys. Returns [P, T] bool: True where key j lies outside the forced region of
    the query at pos, i.e. init_blocks*block <= j < (pos // block - local_blocks) * block. Causality is implied, since
    the upper bound is below pos. A query whose window reaches the init block has no free key (all False)."""
    j = torch.arange(T, device=pos.device).unsqueeze(0)                       # [1, T]
    hi = ((pos // block) - local_blocks).clamp(min=0) * block                  # first forced local token, [P]
    return (j >= init_blocks * block) & (j < hi.unsqueeze(-1))


def forced_blocks(p, block=64, local_blocks=32, init_blocks=1):
    """The forced block indices of a query at token position p (for tests and documentation)."""
    qb = p // block
    return sorted(set(range(init_blocks)) | set(range(max(0, qb - local_blocks), qb + 1)))
