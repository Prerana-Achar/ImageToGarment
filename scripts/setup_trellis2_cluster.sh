#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRELLIS2_DIR="${TRELLIS2_DIR:-$REPO_ROOT/third_party/TRELLIS.2}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
REBUILD_EXTENSIONS="${REBUILD_EXTENSIONS:-0}"
EXT_DIR="${TRELLIS2_EXT_DIR:-${TMPDIR:-/tmp}/trellis2_extensions_${USER:-user}}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found on PATH" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Could not find venv activation script: $VENV_DIR/bin/activate" >&2
  echo "Set VENV_DIR=/path/to/venv if your venv lives elsewhere." >&2
  exit 1
fi

mkdir -p "$(dirname "$TRELLIS2_DIR")"

if [ ! -d "$TRELLIS2_DIR/.git" ]; then
  git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive "$TRELLIS2_DIR"
else
  git -C "$TRELLIS2_DIR" submodule update --init --recursive
fi

cd "$TRELLIS2_DIR"
source "$VENV_DIR/bin/activate"

export CUDA_HOME
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PIP_NO_BUILD_ISOLATION=1

py_has() {
  python - "$1" <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

pip_install_if_missing() {
  local module="$1"
  shift
  if py_has "$module"; then
    echo "[skip] Python module already available: $module"
  else
    python -m pip install "$@"
  fi
}

clone_or_update() {
  local repo_url="$1"
  local dest="$2"
  local ref="${3:-}"
  local extra_clone_args="${4:-}"

  if [ -d "$dest/.git" ]; then
    echo "[reuse] $dest"
    git -C "$dest" fetch --tags --quiet origin || true
  elif [ -e "$dest" ]; then
    echo "Error: $dest exists but is not a git checkout. Move it aside or set TRELLIS2_EXT_DIR to a clean directory." >&2
    exit 1
  else
    mkdir -p "$(dirname "$dest")"
    # shellcheck disable=SC2086
    git clone $extra_clone_args "$repo_url" "$dest"
  fi

  if [ -n "$ref" ]; then
    git -C "$dest" checkout -q "$ref"
    git -C "$dest" submodule update --init --recursive
  fi
}

install_git_extension_if_missing() {
  local module="$1"
  local repo_url="$2"
  local dest="$3"
  local ref="$4"
  local extra_clone_args="${5:-}"

  if py_has "$module"; then
    echo "[skip] Python module already available: $module"
    return
  fi

  if [ -d "$dest" ] && [ "$REBUILD_EXTENSIONS" != "1" ]; then
    echo "[skip] Extension source already exists: $dest"
    echo "       Not rebuilding it. Set REBUILD_EXTENSIONS=1 to force pip install."
    return
  fi

  clone_or_update "$repo_url" "$dest" "$ref" "$extra_clone_args"
  python -m pip install --no-build-isolation "$dest"
}

echo "Using venv: $VENV_DIR"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using extension cache: $EXT_DIR"

python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1" wheel packaging
python -m pip install "numpy==1.26.4"

if ! py_has torch; then
  python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi

# Basic runtime deps. Install only missing top-level modules to avoid churn in this shared venv.
pip_install_if_missing imageio imageio imageio-ffmpeg
pip_install_if_missing tqdm tqdm
pip_install_if_missing easydict easydict
pip_install_if_missing cv2 opencv-python-headless
pip_install_if_missing ninja ninja
pip_install_if_missing trimesh trimesh
pip_install_if_missing transformers transformers
pip_install_if_missing gradio gradio==6.0.1
pip_install_if_missing tensorboard tensorboard
pip_install_if_missing pandas pandas
pip_install_if_missing lpips lpips
pip_install_if_missing zstandard zstandard
pip_install_if_missing kornia kornia
pip_install_if_missing timm timm
pip_install_if_missing scipy scipy
pip_install_if_missing plyfile plyfile
pip_install_if_missing moderngl moderngl

# utils3d has a loose NumPy dependency. Install its deps above, then install it without deps so NumPy stays <2.
if py_has utils3d; then
  echo "[skip] Python module already available: utils3d"
else
  python -m pip install --no-deps git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi
python -m pip install "numpy==1.26.4"

if py_has flash_attn; then
  echo "[skip] Python module already available: flash_attn"
else
  python -m pip install flash-attn==2.7.3 --no-build-isolation
fi

install_git_extension_if_missing nvdiffrast https://github.com/NVlabs/nvdiffrast.git "$EXT_DIR/nvdiffrast" v0.4.0 "-b v0.4.0"
install_git_extension_if_missing renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXT_DIR/nvdiffrec" renderutils "-b renderutils"
install_git_extension_if_missing cumesh https://github.com/JeffreyXiang/CuMesh.git "$EXT_DIR/CuMesh" main "--recursive"
install_git_extension_if_missing flexgemm https://github.com/JeffreyXiang/FlexGEMM.git "$EXT_DIR/FlexGEMM" main "--recursive"

if py_has o_voxel; then
  echo "[skip] Python module already available: o_voxel"
else
  python -m pip install "$TRELLIS2_DIR/o-voxel" --no-build-isolation
fi

python -m pip install "numpy==1.26.4"

echo
echo "TRELLIS.2 setup finished."
echo "Next:"
echo "  cd $TRELLIS2_DIR"
echo "  source $VENV_DIR/bin/activate"
echo "  python example.py"