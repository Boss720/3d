"""Read Windows DXGI adapter ordinals used by ONNX Runtime DirectML.

DXGI ordinals are deliberately never inferred from nvidia-smi: on hybrid
laptops the Intel display adapter commonly precedes the NVIDIA adapter.
No subprocess or hardware enumeration is performed per inference frame.
"""
from __future__ import annotations

import ctypes
import csv
import functools
import io
import os
from pathlib import Path
import subprocess
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


def _normalize_pci_bus_id(value: str) -> str:
    """Join CUDA's four-digit and nvidia-smi's eight-digit PCI domains."""
    try:
        domain, bus, slot = value.strip().lower().split(':')
        device, function = slot.split('.')
        return f'{int(domain, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{int(function, 16):x}'
    except (ValueError, AttributeError):
        return ''


class _CudaProperties(ctypes.Structure):
    # cudaDeviceProp v2 prefix from CUDA 12 driver_types.h. The complete
    # versioned structure is < 2KB; reserve 4KB so the runtime can fill its tail.
    _fields_ = [('name', ctypes.c_char * 256), ('uuid', ctypes.c_ubyte * 16),
                ('luid', ctypes.c_char * 8), ('luid_mask', ctypes.c_uint),
                ('memory', ctypes.c_size_t), ('reserved', ctypes.c_ubyte * 4096)]


def _load_cuda_runtime() -> Any:
    windows = sys.platform == 'win32'
    names = ('cudart64_12.dll',) if windows else ('libcudart.so.12',)
    loader = ctypes.WinDLL if windows else ctypes.CDLL
    roots = [Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)),
             *(Path(path) for path in sys.path if path)]
    candidates = [root / 'nvidia' / 'cuda_runtime' / ('bin' if windows else 'lib') / name
                  for root in roots for name in names]
    for candidate in candidates:
        if candidate.is_file():
            return loader(str(candidate))
    for name in names:
        try:
            return loader(name)
        except OSError:
            pass
    raise RuntimeError('CUDA 12 runtime library is missing. Install the CUDA requirements or use the CUDA packaged build.')


def _cuda_runtime_devices() -> list[dict[str, Any]]:
    # The CUDA runtime, rather than cuDeviceGetCount on Windows, applies
    # CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER used by ONNX Runtime.
    runtime = _load_cuda_runtime()

    def call(name: str, argtypes: list[Any], *args: Any) -> None:
        function = getattr(runtime, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        result = function(*args)
        if result != 0:
            raise RuntimeError(f'{name} failed (CUDA runtime error {result}). Check the NVIDIA driver and CUDA_VISIBLE_DEVICES setting.')

    pointer = ctypes.POINTER(ctypes.c_int)
    version = ctypes.c_int()
    call('cudaRuntimeGetVersion', [pointer], ctypes.byref(version))
    if not 12000 <= version.value < 13000:
        raise RuntimeError(f'Unsupported CUDA runtime version {version.value}; CUDA 12 is required.')
    count = ctypes.c_int()
    call('cudaGetDeviceCount', [pointer], ctypes.byref(count))
    if not 0 <= count.value <= 128:
        raise RuntimeError('CUDA runtime returned an invalid visible device count.')
    devices = []
    for ordinal in range(count.value):
        properties = _CudaProperties()
        call('cudaGetDeviceProperties_v2', [ctypes.c_void_p, ctypes.c_int], ctypes.byref(properties), ordinal)
        major, minor = ctypes.c_int(), ctypes.c_int()
        call('cudaDeviceGetAttribute', [pointer, ctypes.c_int, ctypes.c_int], ctypes.byref(major), 75, ordinal)
        call('cudaDeviceGetAttribute', [pointer, ctypes.c_int, ctypes.c_int], ctypes.byref(minor), 76, ordinal)
        pci = ctypes.create_string_buffer(64)
        call('cudaDeviceGetPCIBusId', [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], pci, len(pci), ordinal)
        if not properties.name or properties.memory == 0 or major.value <= 0:
            raise RuntimeError('CUDA runtime returned invalid device properties.')
        devices.append({'id': ordinal, 'name': properties.name.decode('utf-8', errors='replace'),
                        'memory_mb': round(properties.memory / 1024**2),
                        'compute_capability': [major.value, minor.value],
                        'pci_bus_id': _normalize_pci_bus_id(pci.value.decode('ascii')),
                        'uuid': str(uuid.UUID(bytes=bytes(properties.uuid))), 'driver_model': None})
    return devices


@functools.lru_cache(maxsize=1)
def probe_cuda_devices() -> dict[str, Any]:
    """Enumerate CUDA-visible ordinals; enrich driver mode by physical PCI ID."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        return {'devices': [], 'error': 'No CUDA devices are visible: CUDA_VISIBLE_DEVICES is empty.'}
    try:
        devices = _cuda_runtime_devices()
    except (OSError, AttributeError, RuntimeError, ValueError) as exc:
        return {'devices': [], 'error': f'CUDA device enumeration failed: {str(exc)[:400]}'}
    if not devices:
        return {'devices': [], 'error': 'No CUDA devices are visible. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES setting.'}
    # Optional CLI metadata must never replace CUDA identities or instance VRAM.
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_model.current,pci.bus_id',
             '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True, timeout=5,
        )
        physical = {}
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) == 5:
                index, _, _, driver_model, bus_id = (value.strip() for value in row)
                key = _normalize_pci_bus_id(bus_id)
                if key:
                    physical[key] = {'physical_index': int(index), 'driver_model': driver_model}
        for device in devices:
            device.update(physical.get(device['pci_bus_id'], {}))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {'devices': devices, 'error': None}
