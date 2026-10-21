# PyInstaller one-file build. Build with .venv-gpu so the executable includes
# ONNX Runtime DirectML and its CPU provider. Media models stay downloadable.
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules
from PyInstaller.config import CONF
from pathlib import Path
import json
import sys

sys.path.insert(0, SPECPATH)
from build_identity import build_manifest, MANIFEST_NAME

manifest_path = Path(CONF['workpath']) / MANIFEST_NAME
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(build_manifest(Path(SPECPATH)), sort_keys=True), encoding='utf-8')

datas = [("static", "static"), (str(manifest_path), '.')]
binaries = collect_dynamic_libs("onnxruntime")
hiddenimports = ["onnxruntime.capi.onnxruntime_pybind11_state", "python_multipart"]
hiddenimports += collect_submodules("uvicorn")

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
