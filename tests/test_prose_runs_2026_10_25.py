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
    for exact in ("276.9 s", "172.8 s", "330 s"):
        assert exact not in md, exact
    assert "between 142.8 and 150.7 s of wall time" in md
    assert "`ef5a609a` on 2026-10-25" in md


def test_historical_wording_records_every_hosted_run():
    # Historical wording (2026-10-26): timings are records of the named hosted runs, never a promise; no run-status
    # sentence; the third Colab T4 run (blob 5700491c) is named. Fails on PR #3 head 0910bfe.
    md = _markdown()
    for gone in ("takes about 2½ minutes", "A hosted run of this revision is pending", "exactly what to expect",
                 "(about 1 minute)"):
        assert gone not in md, gone
    for quoted in ("`5700491c` on 2026-10-25: 150.7 s, 45 s, 32.3 s", "Your run may take a different time.",
                   "this can vary between runs"):
        assert quoted in md, quoted
