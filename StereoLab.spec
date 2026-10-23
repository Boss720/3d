# PyInstaller build. DirectML uses the one-file executable; CUDA uses a
# one-folder package so its NVIDIA runtime remains manageable to build/install.
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules
from PyInstaller.config import CONF
from pathlib import Path
import os
import json
import site
import sys

sys.path.insert(0, SPECPATH)
from build_identity import build_manifest, MANIFEST_NAME

manifest_path = Path(CONF['workpath']) / MANIFEST_NAME
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(build_manifest(Path(SPECPATH)), sort_keys=True), encoding='utf-8')

datas = [
    ("static", "static"),
    (str(manifest_path), '.'),
    (str(Path(SPECPATH) / "LICENSE"), '.'),
]
binaries = collect_dynamic_libs("onnxruntime")
hiddenimports = ["onnxruntime.capi.onnxruntime_pybind11_state", "python_multipart"]
hiddenimports += collect_submodules("uvicorn")

cuda_bundle = os.environ.get("STEREOLAB_CUDA_BUNDLE") == "1"
if cuda_bundle:
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA build requires an ONNX Runtime environment with CUDAExecutionProvider.")
    cuda_provider_dll = [source for source, _ in binaries
                         if Path(source).name.lower() == "onnxruntime_providers_cuda.dll"]
    if not cuda_provider_dll:
        raise RuntimeError("The ONNX Runtime CUDA provider DLL was not collected.")

    nvidia_dlls = []
    for packages_dir in site.getsitepackages():
        packages_root = Path(packages_dir)
        nvidia_root = packages_root / "nvidia"
        if nvidia_root.is_dir():
            for dll in nvidia_root.rglob("*.dll"):
                nvidia_dlls.append((dll, dll.relative_to(packages_root).parent))
    found_names = {dll.name.lower() for dll, _ in nvidia_dlls}
    required_prefixes = ("cublas64_", "cudart64_", "cudnn64_")
    missing = [prefix for prefix in required_prefixes
               if not any(name.startswith(prefix) for name in found_names)]
    if missing:
        raise RuntimeError("CUDA build is missing NVIDIA runtime DLLs: " + ", ".join(missing))
    binaries.extend((str(dll), destination.as_posix()) for dll, destination in nvidia_dlls)
    for packages_dir in site.getsitepackages():
        for license_file in Path(packages_dir).glob("nvidia_*.dist-info/licenses/*"):
            if license_file.is_file():
                package_name = license_file.parents[1].name.removesuffix(".dist-info")
                datas.append((str(license_file), f"licenses/nvidia/{package_name}"))
    hiddenimports += collect_submodules("nvidia")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tkinter", "matplotlib", "torch"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
if cuda_bundle:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="StereoLab",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="StereoLab",
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="StereoLab",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
