"""blindheads: find the few heads where a sparse attention selector loses retrieval, and fix them without training.

Code for "Lost in a Few Heads: Diagnosing and Repairing Block Selection in Sparse Attention".

    heads      head sets (which heads to fix on which checkpoint), as small JSON files
    rule       the threshold rule that turns per-head scores into a head set
    probe      the two read-outs of the calibration probe, independent of any model
    infllm_v2  the worked integration: hooks into the InfLLM-V2 release, the fixes, the probe
"""
from .heads import HeadSet, format_heads, load_heads, parse_heads  # noqa: F401
from .rule import threshold_rule, threshold_steps  # noqa: F401

__version__ = "0.1.0"
