"""Read Windows DXGI adapter ordinals used by ONNX Runtime DirectML.

DXGI ordinals are deliberately never inferred from nvidia-smi: on hybrid
laptops the Intel display adapter commonly precedes the NVIDIA adapter.
No subprocess or hardware enumeration is performed per inference frame.
"""
from __future__ import annotations

import ctypes
import functools
import sys
import uuid
from typing import Any


def enumerate_adapters() -> list[dict[str, Any]]:
    """Enumerate IDXGIFactory1::EnumAdapters1, preserving its device IDs."""
    if sys.platform != 'win32':
        raise RuntimeError('DirectML adapter enumeration requires Windows DXGI.')

    class GUID(ctypes.Structure):
        _fields_ = [('data', ctypes.c_ubyte * 16)]

    class LUID(ctypes.Structure):
        _fields_ = [('LowPart', ctypes.c_uint32), ('HighPart', ctypes.c_int32)]

    class AdapterDesc1(ctypes.Structure):
        _fields_ = [
            ('Description', ctypes.c_wchar * 128), ('VendorId', ctypes.c_uint32),
            ('DeviceId', ctypes.c_uint32), ('SubSysId', ctypes.c_uint32), ('Revision', ctypes.c_uint32),
            ('DedicatedVideoMemory', ctypes.c_size_t), ('DedicatedSystemMemory', ctypes.c_size_t),
            ('SharedSystemMemory', ctypes.c_size_t), ('AdapterLuid', LUID), ('Flags', ctypes.c_uint32),
        ]

    def method(instance: ctypes.c_void_p, index: int, result: Any, *args: Any) -> Any:
        vtable = ctypes.cast(instance, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(vtable[index])

    # IDXGIFactory1 IID; factory vtable 12 = EnumAdapters1, adapter vtable 10 = GetDesc1.
    iid = GUID()
    iid.data[:] = uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le
    factory = ctypes.c_void_p()
    dxgi = ctypes.WinDLL('dxgi.dll')
    create = dxgi.CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_int32
    result = create(ctypes.byref(iid), ctypes.byref(factory))
    if result < 0 or not factory:
        raise RuntimeError(f'CreateDXGIFactory1 failed (HRESULT 0x{result & 0xffffffff:08x}).')
    adapters = []
    try:
        enum = method(factory, 12, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        for index in range(128):
            adapter = ctypes.c_void_p()
            result = enum(factory, index, ctypes.byref(adapter))
            if result & 0xffffffff == 0x887A0002:  # DXGI_ERROR_NOT_FOUND
                break
            if result < 0 or not adapter:
                raise RuntimeError(f'EnumAdapters1({index}) failed (HRESULT 0x{result & 0xffffffff:08x}).')
            try:
                desc = AdapterDesc1()
                result = method(adapter, 10, ctypes.c_int32, ctypes.POINTER(AdapterDesc1))(adapter, ctypes.byref(desc))
                if result < 0:
                    raise RuntimeError(f'GetDesc1({index}) failed (HRESULT 0x{result & 0xffffffff:08x}).')
                adapters.append({
                    'id': index, 'name': desc.Description, 'vendor_id': desc.VendorId,
                    'device_id': desc.DeviceId, 'memory_mb': round(desc.DedicatedVideoMemory / 1024**2),
                    'shared_memory_mb': round(desc.SharedSystemMemory / 1024**2),
                    'software': bool(desc.Flags & 2),
                    'luid': f'{desc.AdapterLuid.HighPart & 0xffffffff:08x}:{desc.AdapterLuid.LowPart:08x}',
                })
            finally:
                method(adapter, 2, ctypes.c_uint32)(adapter)
        else:
            raise RuntimeError('DXGI adapter enumeration exceeded the supported adapter count.')
    finally:
        method(factory, 2, ctypes.c_uint32)(factory)
    return adapters


def select_adapter(adapters: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer a physical NVIDIA adapter, then highest dedicated memory."""
    candidates = [item for item in adapters if not item.get('software')]
    nvidia = [item for item in candidates if item.get('vendor_id') == 0x10DE]
    candidates = nvidia or candidates
    return max(candidates, key=lambda item: item.get('memory_mb', 0)) if candidates else None


@functools.lru_cache(maxsize=1)
def probe_hardware() -> dict[str, Any]:
    # The Linux release supports CPU inference. Do not probe Windows APIs or
    # present the absence of DXGI as a broken CPU installation.
    if sys.platform != 'win32':
        return {'adapters': [], 'selected_adapter': None, 'error': None}
    try:
        adapters = enumerate_adapters()
        return {'adapters': adapters, 'selected_adapter': select_adapter(adapters), 'error': None}
    except Exception as exc:
        return {'adapters': [], 'selected_adapter': None, 'error': str(exc)}
