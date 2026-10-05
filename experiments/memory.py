"""Windows OS memory counters and a hard per-process commit limit."""
import ctypes as c
from ctypes import wintypes as w
import os


def memory_snapshot():
    if os.name != 'nt':
        import psutil
        info = psutil.Process().memory_info()
        return dict(working_set=info.rss, peak_working_set=info.rss,
                    private_commit=info.vms, peak_private_commit=info.vms)
    class PMC(c.Structure):
        _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD)] + [(k, c.c_size_t) for k in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
    p = PMC(); p.cb = c.sizeof(p)
    k = c.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = w.HANDLE
    ps = c.WinDLL('psapi', use_last_error=True)
    ps.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(PMC), w.DWORD]
    if not ps.GetProcessMemoryInfo(k.GetCurrentProcess(), c.byref(p), p.cb):
        raise c.WinError(c.get_last_error())
    return dict(working_set=p.WorkingSetSize, peak_working_set=p.PeakWorkingSetSize,
                private_commit=p.PrivateUsage, peak_private_commit=p.PeakPagefileUsage)


class ProcessLimit:
    def __init__(self, process_handle, limit_bytes):
        self.handle = None
        if os.name != 'nt':
            return
        class BASIC(c.Structure):
            _fields_ = [('PerProcessUserTimeLimit', c.c_longlong), ('PerJobUserTimeLimit', c.c_longlong),
                        ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', c.c_size_t),
                        ('MaximumWorkingSetSize', c.c_size_t), ('ActiveProcessLimit', w.DWORD),
                        ('Affinity', c.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(name, c.c_ulonglong) for name in
                        ['ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]
        class EXT(c.Structure):
            _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', IO),
                        ('ProcessMemoryLimit', c.c_size_t), ('JobMemoryLimit', c.c_size_t),
                        ('PeakProcessMemoryUsed', c.c_size_t), ('PeakJobMemoryUsed', c.c_size_t)]
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.k.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.k.CreateJobObjectW.restype = w.HANDLE
        self.k.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.k.CloseHandle.argtypes = [w.HANDLE]
        handle = self.k.CreateJobObjectW(None, None)
        if not handle:
            raise c.WinError(c.get_last_error())
        info = EXT(); info.ProcessMemoryLimit = limit_bytes
        info.BasicLimitInformation.LimitFlags = 0x100 | 0x2000  # PROCESS_MEMORY | KILL_ON_JOB_CLOSE
        if not self.k.SetInformationJobObject(handle, 9, c.byref(info), c.sizeof(info)):
            self.k.CloseHandle(handle); raise c.WinError(c.get_last_error())
        if not self.k.AssignProcessToJobObject(handle, w.HANDLE(int(process_handle))):
            self.k.CloseHandle(handle); raise c.WinError(c.get_last_error())
        self.handle = handle

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None
