"""Frozen, sequential comparison of SOFT_BAYES_COMPLETE.md and released AIOLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

from .soft_bayes_supervisor import ROOT, dump, read, run_one, assert_idle

STUDY = ROOT / "experiments/results/soft_bayes_2026-09-11"
DATA_SEEDS = (911201, 911202)
SAMPLER_SEEDS = (912301, 912302)
SMALL = ((1, 1., 12), (10, 1., 20), (10, 10., 50))
LARGE = ((10, 100., 500), (100, 1000., 500), (1000, 1000., 500))


def hashes(folder):
    result = {}
    for name in ("X.bin", "y.bin", "data.json"):
        h = hashlib.sha256()
        with (Path(folder) / name).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        result[name] = h.hexdigest()
    return result


def prepare():
    assert_idle()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from .data import generate_case
    from .soft_bayes import SoftBayes
    STUDY.mkdir(parents=True, exist_ok=True)
    specs, datasets, preflight = [], {}, []
    for stage, settings in (("prescribed", SMALL), ("practical", LARGE)):
        stage_specs = []
        for d, B, T in settings:
            for data_seed in DATA_SEEDS:
                case_id = f"signal_d{d}_B{B:g}_T{T}_s{data_seed}"
                folder = ROOT / "experiments/data/soft_bayes_2026-09-11" / case_id
                case = dict(d=d, B=B, T=T, family="signal", data_seed=data_seed,
                            B_generator=min(3., B))
                meta = generate_case(folder, case)
                datasets[case_id] = dict(path=str(folder), metadata=meta, hashes=hashes(folder))
                base = dict(d=d, B=B, T=T, rounds=T, case_id=case_id, stage=stage,
                            seconds=20., data_path=str(folder), data_hashes=datasets[case_id]["hashes"])
                choices = [("author_aioli", 0, {})]
                if stage == "prescribed":
                    choices += [("prescribed", s, {"mode": "prescribed", "max_total_moves": None}) for s in SAMPLER_SEEDS]
                else:
                    for samples, transitions in ((16, 1), (64, 4)):
                        choices += [(f"practical_m{samples}n{transitions}", s,
                                     dict(mode="practical", samples=samples, transitions=transitions))
                                    for s in SAMPLER_SEEDS]
                for label, seed, config in choices:
                    run_id = f"{case_id}__{label}__s{seed}"
                    stage_specs.append(dict(base, id=run_id, label=label, seed=seed,
                        algorithm="author_aioli" if label == "author_aioli" else "soft_bayes",
                        config=config, output=str(STUDY / "runs" / run_id)))
            pf = SoftBayes.preflight(d, B, T, {"mode": "prescribed", "max_total_moves": None})
            preflight.append(dict(d=d, B=B, T=T, preflight=pf))
        random.Random(913401).shuffle(stage_specs)
        specs.extend(stage_specs)
    for d, B, T in LARGE:
        case_id = f"signal_d{d}_B{B:g}_T{T}_s{DATA_SEEDS[0]}"
        ds = datasets[case_id]
        run_id = f"{case_id}__prescribed_resource__s{SAMPLER_SEEDS[0]}"
        specs.append(dict(id=run_id, stage="resource", label="prescribed_resource", algorithm="soft_bayes",
                          d=d, B=B, T=T, rounds=T, seconds=10., seed=SAMPLER_SEEDS[0], case_id=case_id,
                          data_path=ds["path"], data_hashes=ds["hashes"],
                          config={"mode": "prescribed", "max_total_moves": None},
                          output=str(STUDY / "runs" / run_id)))
    manifest = dict(schema_version=1, protocol=str(ROOT / "SOFT_BAYES_EXPERIMENT_PLAN.md"),
                    datasets=datasets, runs=specs, preflight=preflight, expected_attempts=51)
    assert len(specs) == 51
    path = STUDY / "manifest.json"
    if path.exists():
        if read(path) != manifest:
            raise RuntimeError("Frozen manifest differs; use a new study")
    else:
        dump(path, manifest)
    print(json.dumps({"prepared_attempts": len(specs), "datasets": len(datasets)}), flush=True)


def run(stages):
    manifest = read(STUDY / "manifest.json")
    for spec in manifest["runs"]:
        if spec["stage"] not in stages:
            continue
        output = Path(spec["output"])
        if (output / "supervisor.json").exists():
            old = read(output / "spec.json")
            if any(old.get(k) != v for k, v in spec.items()):
                raise RuntimeError("Existing attempt configuration changed")
            continue
        if output.exists():
            raise RuntimeError(f"Incomplete attempt directory requires review: {output}")
        if hashes(spec["data_path"]) != spec["data_hashes"]:
            raise RuntimeError("Dataset changed before execution")
        print(json.dumps({"starting": spec["id"]}), flush=True)
        result = run_one(spec)
        if hashes(spec["data_path"]) != spec["data_hashes"]:
            raise RuntimeError("Dataset changed during execution")
        w = result["worker"]
        print(json.dumps(dict(id=spec["id"], status=result["status"],
            rounds=w.get("committed_round", 0), loss=w.get("cumulative_loss"),
            online_seconds=w.get("online_seconds"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", default=["prescribed", "practical", "resource"])
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run(args.stages)


if __name__ == "__main__":
    main()
