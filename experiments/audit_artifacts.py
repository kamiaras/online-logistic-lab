"""Check saved measurement consistency without rerunning any algorithm."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def main():
    base = Path(__file__).resolve().parent
    read = lambda path: json.loads(path.read_text())
    stages = ['calibration', 'prescribed', 'screen', 'confirmation',
              'long_horizon', 'tempered', 'matrix_control']
    checks = {}
    for stage in stages:
        specs = read(base / 'results' / stage / 'manifest.json')
        counts = Counter()
        for spec in specs:
            output = Path(spec['output'])
            assert read(output / 'spec.json') == spec, (stage, output, 'spec')
            result = read(output / 'result.json')
            supervisor = read(output / 'supervisor.json')
            assert supervisor['os_hard_limit_active']
            assert supervisor['hard_process_commit_limit'] == 2 * 1024**3
            assert supervisor['returncode'] == 0, output
            assert result['error'] is None, output
            assert result['threadpools'], output
            assert all(p['num_threads'] == 1 for p in result['threadpools'])
            n = result['round']
            assert 0 < n <= spec['T']
            losses = np.fromfile(output / 'losses.bin', dtype=np.float64)
            times = np.fromfile(output / 'online_times.bin', dtype=np.float64)
            assert len(losses) == len(times) == n, output
            assert np.isfinite(losses).all() and np.isfinite(times).all()
            assert (losses >= 0).all() and (np.diff(times) >= 0).all()
            assert math.isclose(math.fsum(losses), result['cumulative_loss'],
                                rel_tol=1e-11, abs_tol=1e-6), output
            assert math.isclose(times[-1], result['online_seconds'],
                                rel_tol=1e-12, abs_tol=1e-9), output
            if stage == 'confirmation':
                assert n == 500 and result['status'] == 'evaluation_completed'
            counts[result['status']] += 1
        checks[stage] = dict(runs=len(specs), statuses=dict(counts))
    assert sum(c['runs'] for c in checks.values()) == 323
    for stage in ['screen', 'confirmation', 'long_horizon', 'prescribed', 'tempered']:
        for row in read(base / 'results' / stage / 'summary.json'):
            for name in ['common', 'final']:
                value = row.get(name)
                if value:
                    assert value['regret_lower'] <= value['regret_upper']
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(base.glob('*.py'))}
    hashes['requirements.txt'] = hashlib.sha256((base / 'requirements.txt').read_bytes()).hexdigest()
    previous = read(base / 'results' / 'controls.json')['algorithm_source_sha256']
    core = ['aioli.py', 'common.py', 'proximal.py', 'sigmoid.py', 'data.py']
    unchanged = {name: hashes[name] == previous[name] for name in core}
    assert all(unchanged.values())
    (base / 'results' / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    report = dict(passed=True, stages=checks,
                  core_matches_control_snapshot=unchanged,
                  note='Artifact checks only; not a new numerical benchmark or floating-point proof.')
    (base / 'results' / 'artifact_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
