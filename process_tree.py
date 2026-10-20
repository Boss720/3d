"""Read-only Windows process-tree observation helpers.

The pure tree functions are platform-independent. The snapshot function uses the
Windows Toolhelp API and never signals or manages any process.
"""
from __future__ import annotations

import ctypes
import ntpath
import os
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any


class ProcessRootNotFound(LookupError):
    """The supplied root PID could not be validated in this process snapshot."""


def executable_basename(path: str) -> str:
    """Return a case-folded Windows executable basename from either path style."""
    if not isinstance(path, str) or not path.strip():
        return ""
    return ntpath.basename(path.replace("/", "\\")).casefold()


def _valid_pid(value: Any, *, allow_zero: bool = False) -> bool:
    return (type(value) is int and value >= (0 if allow_zero else 1))


def _creation_time_precedes_snapshot(created: Any, snapshot_started: Any) -> bool:
    """Return whether a process identity could belong to a captured row.

    Equality is rejected because FILETIME granularity and clock adjustments can
    make a process created at the snapshot boundary indistinguishable from a
    PID that was reused after the Toolhelp snapshot began.
    """
    return (_valid_pid(created) and _valid_pid(snapshot_started)
            and created < snapshot_started)


def _normalize_entry(entry: Any) -> dict | None:
    if not isinstance(entry, Mapping):
        return None
    pid = entry.get("pid")
    parent_pid = entry.get("parent_pid")
    executable = entry.get("executable")
    if (not _valid_pid(pid) or not _valid_pid(parent_pid, allow_zero=True)
            or not isinstance(executable, str) or not executable.strip()):
        return None

    created = entry.get("creation_time_100ns")
    identity_verified = (entry.get("identity_status") == "verified"
                         and _valid_pid(created))
    normalized = {
        "pid": pid,
        "parent_pid": parent_pid,
        "executable": executable,
        "creation_time_100ns": created if identity_verified else None,
        "identity_status": "verified" if identity_verified else "unavailable",
    }
    identity_error = entry.get("identity_error")
    if not identity_verified and isinstance(identity_error, str) and identity_error:
        normalized["identity_error"] = identity_error
    return normalized


def _normalize_entries(entries: Iterable[Mapping]) -> dict[int, dict]:
    by_pid: dict[int, dict] = {}
    duplicates: set[int] = set()
    for raw in entries:
        entry = _normalize_entry(raw)
        if entry is None:
            continue
        pid = entry["pid"]
        if pid in by_pid:
            duplicates.add(pid)
        else:
            by_pid[pid] = entry
    if duplicates:
        raise ValueError(f"Process snapshot contains duplicate PIDs: {sorted(duplicates)}")
    return by_pid


def find_descendants(entries: Iterable[Mapping], root_pid: int) -> list[dict]:
    """Return validated descendants of ``root_pid`` in stable depth/PID order.

    The root itself is excluded. A missing or malformed root row raises
    :class:`ProcessRootNotFound`, so callers can distinguish a verified empty
    tree from a snapshot that cannot validate the supplied API PID. Malformed
    non-root rows are ignored, and creation-time uncertainty is preserved on
    each returned entry.
    """
    if not _valid_pid(root_pid):
        raise ValueError("Root PID must be a positive integer.")
    by_pid = _normalize_entries(entries)
    if root_pid not in by_pid:
        raise ProcessRootNotFound(f"Root PID {root_pid} was not present in the process snapshot.")

    children: dict[int, list[dict]] = defaultdict(list)
    for entry in by_pid.values():
        if entry["pid"] != root_pid:
            children[entry["parent_pid"]].append(entry)
    for child_entries in children.values():
        child_entries.sort(key=lambda item: item["pid"])

    result = []
    visited = {root_pid}
    queue = deque([(root_pid, 0, by_pid[root_pid]["identity_status"] == "verified")])
    while queue:
        parent_pid, parent_depth, parent_path_verified = queue.popleft()
        parent_entry = by_pid[parent_pid]
        for child in children.get(parent_pid, ()):
            child_pid = child["pid"]
            if child_pid in visited:
                continue
            parent_created = parent_entry["creation_time_100ns"]
            child_created = child["creation_time_100ns"]
            both_identities_verified = (parent_entry["identity_status"] == "verified"
                                         and child["identity_status"] == "verified")
            if both_identities_verified and child_created < parent_created:
                # The child predates the current process using this parent PID.
                # Its recorded parent PID therefore refers to an earlier process.
                continue
            visited.add(child_pid)
            depth = parent_depth + 1
            edge_verified = both_identities_verified
            tree_verified = parent_path_verified and edge_verified
            result.append({**child, "depth": depth,
                           "parent_relationship_status": "verified" if edge_verified else "unverified",
                           "tree_identity_status": "verified" if tree_verified else "unverified"})
            queue.append((child_pid, depth, tree_verified))
    result.sort(key=lambda item: (item["depth"], item["pid"]))
    return result


def _identity_of(entry: Mapping) -> tuple[int, int] | None:
    normalized = _normalize_entry(entry)
    if normalized is None or normalized["identity_status"] != "verified":
        return None
    return normalized["pid"], normalized["creation_time_100ns"]


def match_process_identity(entries: Iterable[Mapping], observed_entry: Mapping) -> dict:
    """Match a prior process by PID and creation FILETIME in a later snapshot.

    ``absent`` means the PID is not present in the current snapshot. When the
    PID is present, an unreadable identity is ``unverified``; a different
    verified FILETIME is ``reused``. This also detects a still-running child
    after it has become an orphan and disappeared from its original tree.
    """
    observed = _normalize_entry(observed_entry)
    if observed is None:
        raise ValueError("Observed process must include a valid PID, parent PID, and executable name.")
    observed_pid = observed["pid"]
    observed_created = observed["creation_time_100ns"]
    observed_identity_verified = observed["identity_status"] == "verified"

    candidates = []
    for raw in entries:
        if not isinstance(raw, Mapping) or raw.get("pid") != observed_pid:
            continue
        candidates.append(raw)
    if not candidates:
        return {"pid": observed_pid,
                "status": "absent" if observed_identity_verified else "unverified",
                "identity_verified": observed_identity_verified,
                "observed_creation_time_100ns": observed_created,
                "current_creation_time_100ns": None,
                **({} if observed_identity_verified else {
                    "reason": "observed creation identity is unavailable; absence cannot be confirmed"})}
    if len(candidates) != 1:
        return {"pid": observed_pid, "status": "unverified", "identity_verified": False,
                "observed_creation_time_100ns": observed_created,
                "current_creation_time_100ns": None,
                "reason": "current snapshot contains duplicate PID rows"}

    current = _normalize_entry(candidates[0])
    if current is None or current["identity_status"] != "verified" or not observed_identity_verified:
        return {"pid": observed_pid, "status": "unverified", "identity_verified": False,
                "observed_creation_time_100ns": observed_created,
                "current_creation_time_100ns": None,
                "reason": candidates[0].get("identity_error") or "current creation identity is unavailable"}
    current_created = current["creation_time_100ns"]
    status = "present" if current_created == observed_created else "reused"
    return {"pid": observed_pid, "status": status, "identity_verified": True,
            "observed_creation_time_100ns": observed_created,
            "current_creation_time_100ns": current_created}


def _windows_creation_time(kernel32, pid: int, wintypes) -> tuple[int | None, str | None]:
    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    process = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        error = ctypes.get_last_error()
        return None, f"OpenProcess failed with Windows error {error}"
    try:
        created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        if not kernel32.GetProcessTimes(process, ctypes.byref(created), ctypes.byref(exited),
                                        ctypes.byref(kernel), ctypes.byref(user)):
            error = ctypes.get_last_error()
            return None, f"GetProcessTimes failed with Windows error {error}"
        filetime = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        if filetime <= 0:
            return None, "GetProcessTimes returned an invalid creation FILETIME"
        return filetime, None
    finally:
        kernel32.CloseHandle(process)


def snapshot_windows_processes() -> list[dict]:
    """Read the Windows process table and creation FILETIMEs without signaling.

    Entries whose creation identity cannot be queried remain in the snapshot
    with ``identity_status='unavailable'`` and an explanatory error. Callers
    must treat same-PID matches with unavailable identity as uncertain.
    """
    if os.name != "nt":
        raise OSError("Windows Toolhelp process snapshots are unavailable on this platform.")

    import ctypes.wintypes as wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    get_system_time = getattr(kernel32, "GetSystemTimePreciseAsFileTime", None)
    if get_system_time is None:
        get_system_time = kernel32.GetSystemTimeAsFileTime
    get_system_time.argtypes = [ctypes.POINTER(FILETIME)]
    get_system_time.restype = None
    snapshot_started = FILETIME()
    get_system_time(ctypes.byref(snapshot_started))
    snapshot_started_100ns = ((int(snapshot_started.dwHighDateTime) << 32)
                              | int(snapshot_started.dwLowDateTime))

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot is None or snapshot == invalid_handle:
        error = ctypes.get_last_error()
        raise OSError(error, f"CreateToolhelp32Snapshot failed with Windows error {error}")

    entries = []
    try:
        native_entry = PROCESSENTRY32W()
        native_entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(native_entry)):
            error = ctypes.get_last_error()
            if error == 18:  # ERROR_NO_MORE_FILES
                return entries
            raise OSError(error, f"Process32FirstW failed with Windows error {error}")
        while True:
            pid = int(native_entry.th32ProcessID)
            parent_pid = int(native_entry.th32ParentProcessID)
            executable = str(native_entry.szExeFile)
            created, identity_error = _windows_creation_time(kernel32, pid, wintypes)
            if created is not None and not _creation_time_precedes_snapshot(
                    created, snapshot_started_100ns):
                created = None
                identity_error = (
                    "Process creation time was not before the process snapshot began; "
                    "PID reuse cannot be ruled out"
                )
            entry = {
                "pid": pid,
                "parent_pid": parent_pid,
                "executable": executable,
                "creation_time_100ns": created,
                "identity_status": "verified" if created is not None else "unavailable",
            }
            if identity_error:
                entry["identity_error"] = identity_error
            entries.append(entry)

            native_entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(native_entry)):
                error = ctypes.get_last_error()
                if error == 18:  # ERROR_NO_MORE_FILES
                    break
                raise OSError(error, f"Process32NextW failed with Windows error {error}")
    finally:
        kernel32.CloseHandle(snapshot)
    return entries
