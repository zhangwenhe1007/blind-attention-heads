"""Head sets: which selection units a fix acts in.

A unit is written "LAYER.HEAD", for example "9.0" for the first key-value head of layer 9. For the block selectors of the
paper (InfLLM-V2, NOSA, DashAttention, NSA, SeerAttention-R) the unit is a key-value head, whose query heads share one
selection. For MoBA it is a query head.

The head sets of every checkpoint in the paper ship in the `heads/` folder of the repository, one JSON file each.
"""
import json
import os
from dataclasses import asdict, dataclass, field

HEADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heads")


def parse_heads(spec):
    """'9.0,16.0,22.0' or ['9.0', ...] or [(9, 0), ...] -> sorted list of (layer, head) with no duplicates."""
    if isinstance(spec, str):
        spec = [s for s in spec.replace(" ", "").split(",") if s]
    out = set()
    for u in spec:
        if isinstance(u, str):
            layer, _, head = u.partition(".")
            if not (layer.isdigit() and head.isdigit()):
                raise ValueError(f"not a LAYER.HEAD unit: {u!r}")
            u = (int(layer), int(head))
        layer, head = int(u[0]), int(u[1])
        if layer < 0 or head < 0:
            raise ValueError(f"negative index in unit {u!r}")
        out.add((layer, head))
    return sorted(out)


def format_heads(heads):
    """[(9, 0), (16, 0)] -> '9.0,16.0'."""
    return ",".join(f"{layer}.{head}" for layer, head in parse_heads(heads))


@dataclass
class HeadSet:
    model: str                               # the checkpoint, as the paper names it
    selector: str                            # the selector design (InfLLM-V2, SeerAttention-R, ...)
    unit: str                                # "key-value head" or "query head"
    heads: list                              # ["9.0", "16.0", "22.0"], in the order of the score when one is given
    total_units: int = 0                     # how many units the checkpoint has
    source: str = ""                         # how the set was chosen
    scores: dict = field(default_factory=dict)   # unit -> integrated fix gain on the calibration prompts, when shipped
    calibration: dict = field(default_factory=dict)   # prompts, share, step, the sweep's curve

    def units(self):
        return parse_heads(self.heads)

    def by_layer(self):
        """{layer: [head, ...]} of the set."""
        out = {}
        for layer, head in self.units():
            out.setdefault(layer, []).append(head)
        return out

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)
            f.write("\n")


def load_heads(name_or_path):
    """A HeadSet from a JSON file, or from the shipped set of that name (`heads/NAME.json`)."""
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(HEADS_DIR, name_or_path if name_or_path.endswith(".json") else name_or_path + ".json")
    if not os.path.exists(path):
        have = sorted(f[:-5] for f in os.listdir(HEADS_DIR) if f.endswith(".json")) if os.path.isdir(HEADS_DIR) else []
        raise FileNotFoundError(f"no head set {name_or_path!r}; shipped sets: {', '.join(have) or 'none'}")
    with open(path) as f:
        d = json.load(f)
    hs = HeadSet(**d)
    parse_heads(hs.heads)                    # validates the unit strings
    return hs
