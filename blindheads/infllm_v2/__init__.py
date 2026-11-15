"""The worked integration: the InfLLM-V2 release and the checkpoints that share its selector.

    model     load a checkpoint with its sparse selector or with full attention, generate, score RULER
    fixes     the training-free fixes and the exact-mass instrument, installed as hooks on the release's own attention
    selection block selection by attention mass, on exact or low-bit keys (what the scorer and the instrument share)
    fix_gain  the calibration probe's integrated fix gain

For openbmb/InfLLM-V2-Long-Sparse-Base (the paper's InfLLM-V2-8B), openbmb/MiniCPM4-8B and openbmb/MiniCPM4.1-8B, which
share one attention class. The last two ship with the selector switched off, and `load` switches it on.
"""
from .fixes import exact_mass, full_attention_heads, larger_budget, low_bit_scorer  # noqa: F401
from .model import evaluate, generate, load  # noqa: F401
