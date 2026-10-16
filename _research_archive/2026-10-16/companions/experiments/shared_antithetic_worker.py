"""Reuse the unchanged isolated recorder with one initialization-only substitution."""
import sys
if __name__ == "__main__" and "--gate" in sys.argv:
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("Missing supervisor gate")
import json
from pathlib import Path
from . import shared_ablation_worker
from .shared_antithetic_ablation import AntitheticSharedAblation


if __name__ == "__main__":
    shared_ablation_worker.SharedAblation = AntitheticSharedAblation
    shared_ablation_worker.run(json.loads(Path(sys.argv[1]).read_text()))
