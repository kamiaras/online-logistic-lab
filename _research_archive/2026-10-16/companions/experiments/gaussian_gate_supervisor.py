"""Sequential single-worker supervisor with actual-PID hard memory limits.

Use run_one(spec) from a campaign, or python -m this_module spec.json. Numerical
libraries are imported only in the child after its 2 GiB cap is verified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import psutil

from .worker_job import attach_worker

ROOT = Path(__file__).resolve().parents[1]
CAP = 2 * 1024**3
SOURCE_PATHS = (
    "experiments/gaussian_gate_worker.py", "experiments/gaussian_gate_supervisor.py",
    "experiments/gaussian_feature_gate.py", "experiments/author_baselines.py",
    "experiments/aioli.py", "experiments/sigmoid.py", "experiments/common.py",
    "experiments/data.py", "experiments/memory.py", "experiments/worker_job.py",
    "research/author_code_comparison_2026-09-11/a9a_experiment.py",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def assert_idle():
    """Refuse an overlapping benchmark, including prior campaign workers."""
    own_tree = {os.getpid(), *(parent.pid for parent in psutil.Process().parents())}
    active = []
    for process in psutil.process_iter(["pid", "cmdline", "name"]):
        if process.pid in own_tree:
            continue
        try:
            words = process.info["cmdline"] or []
            if any(word.startswith("experiments.") and any(token in word for token in
                       ("worker", "runner", "campaign", "supervisor")) for word in words):
                active.append(process.pid)
            elif not words and "python" in (process.info["name"] or "").lower():
                active.append(process.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if "python" in (process.info["name"] or "").lower():
                active.append(process.pid)
    if active:
        raise RuntimeError("Other experiment processes must finish first: " + str(active))


class _ExclusiveBenchmark:
    """An OS mutex prevents two new supervisors from racing past assert_idle."""
    def __enter__(self):
        if os.name != "nt":
            raise RuntimeError("This benchmark requires Windows Job Objects")
        import ctypes
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        suffix = hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:24]
        self.handle = self.kernel.CreateMutexW(None, False, "Local\\ols2_gaussian_gate_" + suffix)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        outcome = self.kernel.WaitForSingleObject(self.handle, 0)
        if outcome not in (0, 0x80):
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise RuntimeError("Another Gaussian benchmark supervisor holds the worker mutex")
        return self

    def __exit__(self, *_):
        self.kernel.ReleaseMutex(self.handle)
        self.kernel.CloseHandle(self.handle)


def run_one(spec):
    """Run a fresh, non-overlapping attempt and return its terminal supervisor record."""
    spec = dict(spec)
    seconds = float(spec["seconds"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive finite")
    if spec.get("memory_bytes", CAP) != CAP:
        raise ValueError("Every worker must use the hard 2 GiB cap")
    spec["memory_bytes"] = CAP
    spec["output"] = str(Path(spec["output"]).resolve())
    spec["data_path"] = str(Path(spec["data_path"]).resolve())
    if "stop_path" in spec:
        spec["stop_path"] = str(Path(spec["stop_path"]).resolve())
    with _ExclusiveBenchmark():
        assert_idle()
        return _run_locked(spec)


def _run_locked(spec):
    output = Path(spec["output"])
    before = source_hashes()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "spec.json", spec)
    dump(output / "source_hashes_before.json", before)
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    start, proc, job, reason, peak, peak_rss = perf_counter(), None, None, None, 0, 0
    hard, actual_pid, verification = False, None, None
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "experiments.gaussian_gate_worker", str(output / "spec.json"), "--gate"],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW)
            ready = output / "worker_ready.json"
            while actual_pid is None:
                try:
                    actual_pid = int(read(ready)["pid"])
                except (FileNotFoundError, json.JSONDecodeError):
                    if proc.poll() is not None or perf_counter() - start > 20.:
                        raise RuntimeError("Actual worker failed before PID handshake")
                    sleep(.05)
            monitor = psutil.Process(actual_pid)
            if actual_pid != proc.pid and proc.pid not in {p.pid for p in monitor.parents()}:
                raise RuntimeError("Ready PID is not the launched worker or its descendant")
            job = attach_worker(actual_pid, CAP)
            hard = job.handle is not None
            if not hard:
                raise RuntimeError("Hard memory cap was not attached to actual worker")
            proc.stdin.write("GO\n")
            proc.stdin.flush()
            proc.stdin.close()
            while proc.poll() is None:
                if verification is None:
                    try:
                        verification = read(output / "job_verification.json")
                        if (verification["pid"] != actual_pid or
                                verification["process_memory_limit"] != CAP or
                                not verification["verified_inside_actual_interpreter"]):
                            raise RuntimeError("Actual interpreter returned invalid memory-cap verification")
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                try:
                    info = monitor.memory_info()
                    peak = max(peak, getattr(info, "private", info.vms))
                    peak_rss = max(peak_rss, info.rss)
                except psutil.NoSuchProcess:
                    pass
                if perf_counter() - start > spec["seconds"] + 30.:
                    reason = "external_wall_budget"
                elif peak > CAP:
                    reason = "memory_limit"
                if reason:
                    # Closing the actual interpreter's job kills it even if
                    # Popen refers to a Windows venv launcher.
                    job.close()
                    if proc.poll() is None:
                        proc.kill()
                    break
                sleep(.1)
            proc.wait(timeout=10.)
        except Exception as exc:
            reason = "supervisor_error: " + repr(exc)
        finally:
            if job is not None:
                job.close()
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10.)
    wall = perf_counter() - start
    final_available = (output / "result.json").exists()
    worker = read(output / "result.json") if final_available else (
        read(output / "progress.json") if (output / "progress.json").exists() else {})
    if verification is None and (output / "job_verification.json").exists():
        verification = read(output / "job_verification.json")
    if verification is None and reason is None:
        reason = "missing_actual_worker_job_verification"
    elif verification is not None and (
            verification.get("pid") != actual_pid or
            verification.get("process_memory_limit") != CAP or
            not verification.get("verified_inside_actual_interpreter")):
        reason = "invalid_actual_worker_job_verification"
    after = source_hashes()
    dump(output / "source_hashes_after.json", after)
    if before != after:
        reason = "source_identity_changed"
    if proc is not None and proc.returncode != 0 and reason is None:
        reason = "abnormal_exit"
    status = reason or (worker.get("status") if final_available else "abnormal_exit")
    result = dict(
        status=status, termination=reason, returncode=proc.returncode if proc else None,
        actual_worker_pid=actual_pid, launcher_pid=proc.pid if proc else None,
        process_wall_seconds=wall, external_wall_budget_seconds=spec["seconds"] + 30.,
        monitored_peak_private=peak, monitored_peak_rss=peak_rss,
        hard_process_commit_limit=CAP, os_hard_limit_active=hard,
        job_verification=verification, worker_final_available=final_available,
        checkpoint_is_censored=not final_available or reason is not None,
        measurement_status="final_snapshot" if final_available and reason is None else (
            "checkpoint_censored" if worker else "unavailable"),
        sources_unchanged=before == after, workers_at_once=1, blas_threads=1, worker=worker)
    dump(output / "supervisor.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    result = run_one(read(args.spec))
    print(json.dumps({"status": result["status"],
                      "round": result["worker"].get("committed_round", 0)}))


if __name__ == "__main__":
    main()
