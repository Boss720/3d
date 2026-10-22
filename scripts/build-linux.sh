#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != Linux ]]; then
    echo "Build this artifact on Linux; PyInstaller does not cross-compile." >&2
    exit 2
fi
if [[ "$(uname -m)" != x86_64 ]]; then
    echo "Only Linux x86_64 builds are configured; found $(uname -m)." >&2
    exit 2
fi

PYTHON="${STEREOLAB_PYTHON:-python3}"
BUILD_ROOT="${STEREOLAB_BUILD_ROOT:-$ROOT/.runtime/linux-build}"
if [[ "$BUILD_ROOT" != /* ]]; then
    BUILD_ROOT="$ROOT/$BUILD_ROOT"
fi
VENV="$BUILD_ROOT/venv"
DIST="$BUILD_ROOT/dist"
WORK="$BUILD_ROOT/work"
RELEASE="$BUILD_ROOT/release"

mkdir -p "$BUILD_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV"
fi
BUILD_PYTHON="$VENV/bin/python"
"$BUILD_PYTHON" -m pip install --upgrade pip
"$BUILD_PYTHON" -m pip install -r requirements-linux-build.txt
"$BUILD_PYTHON" -m pip check

"$BUILD_PYTHON" -m PyInstaller --clean --noconfirm \
    --distpath "$DIST" --workpath "$WORK" StereoLab-linux.spec

PACKAGE="$DIST/StereoLab"
if [[ ! -x "$PACKAGE/StereoLab" ]]; then
    echo "PyInstaller did not produce an executable Linux package at $PACKAGE." >&2
    exit 1
fi

cat > "$PACKAGE/stereolab" <<'LAUNCHER'
#!/usr/bin/env sh
set -eu
APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$APP_DIR/StereoLab" "$@"
LAUNCHER
chmod 0755 "$PACKAGE/stereolab"
cp "$ROOT/LINUX.md" "$PACKAGE/README-Linux.md"

VERSION="${STEREOLAB_VERSION:-}"
if [[ -z "$VERSION" ]]; then
    VERSION="dev-$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"
fi
if [[ ! "$VERSION" =~ ^[A-Za-z0-9._+-]+$ ]]; then
    echo "STEREOLAB_VERSION contains unsupported characters." >&2
    exit 2
fi

"$BUILD_PYTHON" - "$PACKAGE/build-report.json" "$ROOT" "$VERSION" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path
root = Path(sys.argv[2])
sys.path.insert(0, str(root))
from build_identity import build_fingerprint
def git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
report = {
    'version': sys.argv[3],
    'architecture': platform.machine(),
    'platform': platform.platform(),
    'python': platform.python_version(),
    'build_fingerprint': build_fingerprint(root),
    'source_commit': git('rev-parse', 'HEAD'),
    'working_tree_dirty': bool(git('status', '--porcelain')) if git('rev-parse', '--is-inside-work-tree') else None,
}
(Path(sys.argv[1])).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(report, sort_keys=True))
PY
"$BUILD_PYTHON" -m pip freeze --all > "$PACKAGE/requirements-resolved.txt"
"$BUILD_PYTHON" scripts/verify_linux_elf.py "$PACKAGE" --report "$PACKAGE/native-dependencies.json"
"$BUILD_PYTHON" scripts/collect_linux_notices.py --output "$PACKAGE/third-party-notices"

mkdir -p "$RELEASE"
ARCHIVE="$RELEASE/StereoLab-$VERSION-linux-x86_64-cpu.tar.gz"
tar -czf "$ARCHIVE" -C "$DIST" StereoLab
(cd "$RELEASE" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
cat "$ARCHIVE.sha256"
printf 'Linux package: %s\nBuild report: %s\n' "$ARCHIVE" "$PACKAGE/build-report.json"
