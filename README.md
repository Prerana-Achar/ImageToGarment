# ImageToGarment / ChatGarment Setup

This is the complete initial setup guide for running ChatGarment with
GarmentCodeRC on the cluster.

It assumes:

```text
Linux cluster
Python 3.10
CUDA toolkit 11.8
NVIDIA A100/H100-class GPU for inference
```

The known-good Python stack is:

```text
torch==2.1.2+cu118
torchvision==0.16.2+cu118
flash-attn==2.5.8 built from source
setuptools==69.5.1
```

Do not commit the virtual environment, checkpoints, logs, or generated results.
They are machine-specific and/or large.

## 1. Choose Install Paths

Set these variables first. Change `PROJECT_ROOT` if you want a different
location.

```bash
export PROJECT_ROOT=/is/cluster/pachar/Projects/ImageToGarment
export CHATGARMENT_ROOT=$PROJECT_ROOT/ChatGarment
export GARMENTCODE_ROOT=$PROJECT_ROOT/GarmentCodeRC
export VENV_ROOT=$PROJECT_ROOT/venv
```

## 2. Load Cluster Modules

Use your cluster's Python 3.10 and CUDA 11.8 modules. Example:

```bash
module load cuda/11.8
module load python/3.10
```

Check CUDA:

```bash
nvcc --version
```

The output should report CUDA 11.8.

## 3. Clone Repositories

```bash
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git clone https://github.com/biansy000/ChatGarment.git
git clone https://github.com/biansy000/GarmentCodeRC.git
```

Expected layout:

```text
ImageToGarment/
  ChatGarment/
  GarmentCodeRC/
  venv/
```

## 4. Create Virtual Environment

```bash
cd "$PROJECT_ROOT"
python3.10 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
```

Install build tools. Keep `setuptools==69.5.1`; newer versions can remove
`pkg_resources`, which `torch==2.1.2` still imports while building CUDA
extensions.

```bash
pip install --upgrade pip
pip install --force-reinstall "setuptools==69.5.1"
pip install packaging wheel ninja
```

Verify:

```bash
python -c "import pkg_resources; print('pkg_resources ok')"
```

## 5. Install PyTorch for CUDA 11.8

Install the CUDA 11.8 wheels explicitly:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

Verify Torch and CUDA match:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

Expected:

```text
2.1.2+cu118 11.8 True
```

If Torch reports CUDA 12.1 but `nvcc` reports CUDA 11.8, clean the wrong install:

```bash
pip uninstall -y torch torchvision torchaudio triton \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

## 6. Install ChatGarment

Install ChatGarment without letting pip replace the known-good Torch build:

```bash
cd "$CHATGARMENT_ROOT"
pip install -e ".[train]" --no-deps
```

Install ChatGarment runtime dependencies:

```bash
pip install \
  transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 shortuuid \
  accelerate==0.32.0 peft==0.10.0 bitsandbytes \
  pydantic "markdown2[all]" numpy scikit-learn==1.2.2 \
  gradio==4.16.0 gradio_client==0.8.1 \
  requests httpx uvicorn fastapi \
  einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 \
  opencv-python easydict tensorboard \
  deepspeed==0.12.6 wandb
```

## 7. Build FlashAttention From Source

Use `flash-attn==2.5.8`. Do not install unpinned `flash-attn`; newer releases can
be incompatible with `torch==2.1.2`.

```bash
pip uninstall -y flash-attn
pip install --force-reinstall "setuptools==69.5.1"
pip install packaging wheel ninja

export MAX_JOBS=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE
pip install "flash-attn==2.5.8" --no-build-isolation --no-cache-dir
```

If your job has enough CPU/RAM, you can speed up the build:

```bash
export MAX_JOBS=4
```

Verify:

```bash
python -c "import flash_attn; print('flash-attn ok')"
```

## 8. Install GarmentCodeRC

```bash
cd "$GARMENTCODE_ROOT"
pip install -e .
```

This installs the core `pygarment` dependencies:

```text
pyyaml
numpy<2
scipy
svgwrite
svgpathtools
psutil
matplotlib
CairoSVG
nicegui
trimesh
libigl
pyrender
cgal
```

Verify:

```bash
python -c "import pygarment; print('pygarment ok')"
```

For full simulation support, GarmentCodeRC also expects the custom Warp fork:

```bash
cd "$PROJECT_ROOT"
git clone https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode.git
cd NvidiaWarp-GarmentCode
python build_lib.py
pip install -e .
```

## 9. Create GarmentCodeRC system.json

Create the local GarmentCodeRC config:

```bash
cd "$GARMENTCODE_ROOT"
mkdir -p Logs

cat > system.json <<'EOF'
{
  "output": "./Logs/",
  "datasets_path": "",
  "datasets_sim": "",
  "sim_configs_path": "./assets/Sim_props",
  "bodies_default_path": "./assets/bodies",
  "body_samples_path": ""
}
EOF
```

## 10. Link GarmentCodeRC Assets Into ChatGarment

```bash
cd "$CHATGARMENT_ROOT"
if [ -e assets ] && [ ! -L assets ]; then
  mv assets "assets.backup.$(date '+%Y%m%d_%H%M%S')"
fi
ln -sfn "$GARMENTCODE_ROOT/assets" assets
```

Check:

```bash
ls -l assets
```

## 11. Patch Hardcoded Upstream Paths

The upstream code contains the authors' old local paths. Patch them to your
install location:

```bash
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
import os

project = Path(os.environ["PROJECT_ROOT"])
chat = Path(os.environ["CHATGARMENT_ROOT"])
garment = Path(os.environ["GARMENTCODE_ROOT"])

replacements = {
    "/is/cluster/fast/sbian/github/chatgarment_private": str(chat),
    "/is/cluster/fast/sbian/github/GarmentCodeV2/": str(garment) + "/",
    "/is/cluster/fast/sbian/github/GarmentCodeV2": str(garment),
}

for rel in [
    "ChatGarment/llava/garment_utils_v2.py",
    "ChatGarment/run_garmentcode_sim.py",
]:
    path = project / rel
    text = path.read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched {path}")
PY
```

Verify no old paths remain:

```bash
grep -R "/is/cluster/fast/sbian/github" -n "$CHATGARMENT_ROOT" || true
```

## 12. Add the ChatGarment Checkpoint

Download the pretrained ChatGarment weights from the upstream install docs:

```text
https://sjtueducn-my.sharepoint.com/:u:/g/personal/biansiyuan_sjtu_edu_cn/EQayoB8ie7ZIsFrjLWdBASQBFexZHXcGjrS6ghgGCjIMzw?e=o60Y65
```

Place the downloaded file here:

```text
$CHATGARMENT_ROOT/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin
```

Create the folder:

```bash
mkdir -p "$CHATGARMENT_ROOT/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final"
```

If you download the file on your local machine, copy it to that path. The file
must be named:

```text
pytorch_model.bin
```

Check it is a binary checkpoint and not an HTML download page:

```bash
cd "$CHATGARMENT_ROOT"
file checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin
ls -lh checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin
```

The checkpoint used in this setup was:

```text
691,679,232 bytes
```

## 13. Final Verification

Run:

```bash
cd "$CHATGARMENT_ROOT"
source "$VENV_ROOT/bin/activate"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import pkg_resources; print('pkg_resources ok')"
python -c "import flash_attn; print('flash-attn ok')"
python -c "import llava; print('llava ok')"
python -c "import pygarment; print('pygarment ok')"
```

Expected:

```text
2.1.2+cu118 11.8 True
pkg_resources ok
flash-attn ok
llava ok
pygarment ok
```

## 14. Optional: OpenAI API Key for Text Generation

Image reconstruction does not require an OpenAI key. Text generation/editing
scripts use GPT-4o for prompt rewriting, so set:

```bash
export OPENAI_API_KEY=sk-...
```

## 15. Optional: Submit Batch Image Jobs With Condor

This workspace includes a Condor submitter for categorized images:

```bash
cd "$CHATGARMENT_ROOT"
bash run/submit_garmentimage_batch.sh garmentimage_all
```

Run one category:

```bash
bash run/submit_garmentimage_batch.sh bottoms Bottoms
```

Run selected categories:

```bash
bash run/submit_garmentimage_batch.sh garments Tops Bottoms Skirts Dresses
```

The submitter requests:

```text
1 GPU
>= 80 GB GPU memory
16 CPUs
96 GB RAM
80 GB disk
```

Logs go to:

```text
ChatGarment/cluster_logs/
```

Results go to:

```text
ChatGarment/ChatGarmentResults/
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'packaging'`

```bash
pip install packaging wheel ninja
```

### `ModuleNotFoundError: No module named 'pkg_resources'`

```bash
pip install --force-reinstall "setuptools==69.5.1"
```

### CUDA mismatch during FlashAttention build

Error:

```text
The detected CUDA version (11.8) mismatches the version that was used to compile
PyTorch (12.1).
```

Fix:

```bash
pip uninstall -y torch torchvision torchaudio triton \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### `std::optional` / `c10::optional` FlashAttention C++ errors

You are building a FlashAttention release that is too new for `torch==2.1.2`.
Use:

```bash
export MAX_JOBS=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE
pip install "flash-attn==2.5.8" --no-build-isolation --no-cache-dir
```

### `ModuleNotFoundError: No module named 'pygarment'`

Install GarmentCodeRC into the active venv:

```bash
cd "$GARMENTCODE_ROOT"
pip install -e .
```

If needed, also export:

```bash
export PYTHONPATH="$CHATGARMENT_ROOT:$GARMENTCODE_ROOT:$GARMENTCODE_ROOT/pygarment:$PYTHONPATH"
```

### Checkpoint downloaded as HTML

If this says HTML:

```bash
file "$CHATGARMENT_ROOT/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin"
```

then SharePoint blocked the command-line download. Download it in a browser and
copy the real binary file to the checkpoint path.
