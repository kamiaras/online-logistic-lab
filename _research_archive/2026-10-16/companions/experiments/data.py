"""Fixed synthetic sequences; comparator and generating radii are separate."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .common import sigmoid


def normalized(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-300)


def generate_case(folder, case):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'data.json').exists():
        old = json.loads((folder / 'data.json').read_text())
        if old['case'] != case:
            raise ValueError('Existing data case differs')
        return old
    rng = np.random.default_rng(case['data_seed'])
    d, T, kind = case['d'], case['T'], case['family']
    truth = float(case.get('B_generator', 3.))
    v = normalized(rng.normal(size=d))
    v2 = normalized(rng.normal(size=d))
    hasher = hashlib.sha256()
    norm_min, norm_max, score_sq, label_pos = 2., 0., 0., 0
    with (folder / 'X.bin').open('wb') as fx, (folder / 'y.bin').open('wb') as fy:
        for start in range(0, T, 128):
            n = min(128, T-start)
            if kind == 'coordinate':
                x = np.zeros((n, d))
                # Visit each coordinate in blocks; the truth has norm B_generator.
                idx = (np.arange(start, start+n) // 16) % d
                x[np.arange(n), idx] = rng.choice([-1., 1.], n)
                score = truth * (x @ v)
            else:
                z = rng.normal(size=(n, d))
                if kind in ('signal', 'separable', 'rare_flip', 'shift') and d > 1:
                    z = normalized(z - (z @ v)[:, None] * v)
                    signs = rng.choice([-1., 1.], n)
                    rho = float(case.get('rho', .5))
                    x = rho * signs[:, None] * v + np.sqrt(1-rho*rho) * z
                elif kind == 'correlated':
                    x = normalized(z * np.geomspace(1., .001, d))
                else:
                    x = normalized(z)
                score = truth * (x @ v)
                if kind == 'shift':
                    mask = np.arange(start, start+n) >= T//2
                    score[mask] = truth * (x[mask] @ v2)
            if kind in ('separable', 'rare_flip'):
                y = np.where(score >= 0, 1, -1)
                if kind == 'rare_flip':
                    y = y * np.where(rng.random(n) < .01, -1, 1)
            elif kind == 'random':
                y = rng.choice([-1, 1], n)
            else:
                y = np.where(rng.random(n) < sigmoid(score), 1, -1)
            # Ensure feature bound despite floating-point normalization error.
            x /= np.maximum(1., np.linalg.norm(x, axis=1))[:, None]
            xb, yb = np.ascontiguousarray(x, dtype='<f8').tobytes(), np.asarray(y, dtype='i1').tobytes()
            fx.write(xb); fy.write(yb)
            hasher.update(xb); hasher.update(yb)
            norms = np.linalg.norm(x, axis=1)
            norm_min, norm_max = min(norm_min, float(norms.min())), max(norm_max, float(norms.max()))
            score_sq += float(score @ score)
            label_pos += int(np.sum(y == 1))
    meta = dict(case=case, sha256_interleaved_blocks=hasher.hexdigest(), block_size=128,
                feature_norm_min=norm_min, feature_norm_max=norm_max,
                generating_score_rms=np.sqrt(score_sq/T), positive_fraction=label_pos/T,
                disk_bytes=T*(8*d+1), dtype_X='little-endian float64', dtype_y='int8')
    (folder / 'data.json').write_text(json.dumps(meta, indent=2))
    return meta


def stream_case(folder, d, T):
    with (Path(folder)/'X.bin').open('rb') as fx, (Path(folder)/'y.bin').open('rb') as fy:
        for start in range(0, T, 128):
            n = min(128, T-start)
            x = np.fromfile(fx, dtype='<f8', count=n*d).reshape(n,d)
            y = np.fromfile(fy, dtype='i1', count=n)
            if len(y) != n:
                raise EOFError('Incomplete data')
            yield x, y
