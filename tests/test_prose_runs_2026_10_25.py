"""Prose numbers stay true across every recorded hosted run (Kurt, 2026-10-25: point to output + ranges).

Hosted Colab T4 runs: blob `042c8999` (2026-10-04: 144.5 s wall, setup 56 s, fine-tune 32.0 s) and blob
`ef5a609a` (2026-10-25: 142.8 s, 62 s, 31.9 s). Timings are approximate with the runtime named.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "convnextv2_classification_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_timings_are_approximate_and_name_the_runtime():
    md = _markdown()
    for exact in ("144.5 s", "32.0 s", "setup 56 s", "276.9 s", "172.8 s", "330 s"):
        assert exact not in md, exact
    assert "about 2½ minutes of wall time on a fresh Colab Tesla T4" in md
    assert "`ef5a609a` on 2026-10-25" in md
