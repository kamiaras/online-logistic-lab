"""Attach and verify the actual interpreter, not a Windows venv launcher."""
import ctypes as c
from ctypes import wintypes as w
import os

from .memory import ProcessLimit


def attach_worker(pid, limit):
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE
    if not handle:
        raise c.WinError(c.get_last_error())
    try:
        return ProcessLimit(handle, limit)
    finally:
        kernel.CloseHandle(handle)


def verify_current_limit(expected):
    if os.name != "nt":
        raise RuntimeError("The verified memory test requires Windows")
    class Basic(c.Structure):
        _fields_ = [("PerProcessUserTimeLimit", c.c_longlong), ("PerJobUserTimeLimit", c.c_longlong),
                    ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                    ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]
    class IO(c.Structure):
        _fields_ = [(name, c.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
    class Extended(c.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO), ("ProcessMemoryLimit", c.c_size_t),
                    ("JobMemoryLimit", c.c_size_t), ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    kernel.QueryInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.POINTER(w.DWORD)]
    member, info = w.BOOL(), Extended()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, c.byref(member)):
        raise c.WinError(c.get_last_error())
    if not member.value or not kernel.QueryInformationJobObject(None, 9, c.byref(info), c.sizeof(info), None):
        raise RuntimeError("Actual worker has no queryable memory-limiting Job Object")
    flags = info.BasicLimitInformation.LimitFlags
    if not flags & 0x100 or info.ProcessMemoryLimit != expected:
        raise RuntimeError(f"Actual worker memory cap differs: flags={flags}, bytes={info.ProcessMemoryLimit}")
    return dict(pid=os.getpid(), job_member=True, process_memory_limit=int(info.ProcessMemoryLimit),
                limit_flags=int(flags), verified_inside_actual_interpreter=True)
