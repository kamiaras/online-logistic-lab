"""Complete a runtime-declared primary subset without modifying the original campaign."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
from .shared_ablation_campaign import OUT, ROOT, run_one, source_hashes


def main():
    specs = json.loads((OUT / "manifest.json").read_text())
    targets = [s for s in specs if s["stage"] == "primary" and
               (s["d"] == 10 or s["seed"] <= 910003)]
    assert len(targets) == 36
    assert source_hashes() == json.loads((OUT / "source_hashes_before.json").read_text())
    paths = ["SHARED_ABLATION_AMENDMENT.md", "experiments/shared_ablation_continue.py",
             "experiments/shared_ablation.py", "experiments/shared_ablation_worker.py",
             "experiments/shared_ablation_campaign.py", "experiments/common.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    mp = OUT / "continuation_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps(dict(targets=targets, source_hashes=hashes,
            completed_at_start=[Path(s["output"]).name for s in specs
                if (Path(s["output"]) / "supervisor.json").exists()]), indent=2))
    else:
        assert json.loads(mp.read_text())["targets"] == targets
    start = perf_counter()
    attempted = []
    for i, spec in enumerate(targets):
        if (Path(spec["output"]) / "supervisor.json").exists():
            continue
        if perf_counter() - start >= 1200:
            print("Continuation wall budget reached", flush=True)
            break
        result = run_one(spec)
        rp = Path(spec["output"]) / "result.json"
        worker = json.loads(rp.read_text()) if rp.exists() else {}
        row = dict(index=i+1, total=len(targets), name=Path(spec["output"]).name,
                   status=worker.get("status", "abnormal_exit"), round=worker.get("round", 0),
                   seconds=round(result["wall_seconds"], 3))
        attempted.append(row)
        print(json.dumps(row), flush=True)
    assert source_hashes() == json.loads((OUT / "source_hashes_before.json").read_text())
    assert hashes == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    (OUT / "continuation_result.json").write_text(json.dumps(dict(
        wall_seconds=perf_counter()-start, attempted=attempted,
        missing=[Path(s["output"]).name for s in targets
                 if not (Path(s["output"]) / "supervisor.json").exists()],
        source_hashes_after=hashes), indent=2))


if __name__ == "__main__":
    main()
