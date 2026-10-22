"""Linux x86-64 CPU one-directory build; create it on Linux with requirements.txt."""
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules
from pathlib import Path
import json
import sys

ROOT = Path(SPECPATH).resolve()
sys.path.insert(0, str(ROOT))
from build_identity import MANIFEST_NAME, build_manifest

manifest_path = Path(CONF['workpath']) / MANIFEST_NAME
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(build_manifest(ROOT), sort_keys=True), encoding='utf-8')

datas = [
    (str(ROOT / 'static'), 'static'),
    (str(manifest_path), '.'),
    (str(ROOT / 'LICENSE'), '.'),
]
binaries = collect_dynamic_libs('onnxruntime')
hiddenimports = ['onnxruntime.capi.onnxruntime_pybind11_state', 'python_multipart']
hiddenimports += collect_submodules('uvicorn')

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'tkinter', 'matplotlib', 'torch'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StereoLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='StereoLab',
)
