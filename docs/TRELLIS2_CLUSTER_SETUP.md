# TRELLIS.2 Cluster Setup With Existing venv

This repo can use Microsoft TRELLIS.2 as an external image-to-3D generator. TRELLIS.2 is not vendored here because it has large CUDA dependencies and model weights.

The commands below use the existing repository `venv`; they do not create a conda environment.

## Requirements

- Linux cluster node
- NVIDIA GPU with at least 24 GB VRAM; TRELLIS.2 is verified upstream on A100 and H100 GPUs
- CUDA Toolkit 12.4 recommended
- Existing Python venv at `venv/`
- Git with submodule support
- Hugging Face access for `microsoft/TRELLIS.2-4B`

## One-command setup

From the root of this repository on the cluster:

```bash
bash scripts/setup_trellis2_cluster.sh
```

By default this clones TRELLIS.2 into `third_party/TRELLIS.2`, activates `venv/bin/activate`, checks which Python/CUDA modules are already importable, and installs only missing pieces. It avoids upstream `--basic`, `sudo apt install libjpeg-dev`, flash-attn build isolation, and the non-idempotent `/tmp/extensions` clone path.

You can override paths:

```bash
VENV_DIR=/path/to/venv TRELLIS2_DIR=/path/to/TRELLIS.2 TRELLIS2_EXT_DIR=/path/to/ext-cache CUDA_HOME=/usr/local/cuda-12.4 bash scripts/setup_trellis2_cluster.sh
```

## Manual setup

```bash
mkdir -p third_party
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive third_party/TRELLIS.2
cd third_party/TRELLIS.2

source ../../venv/bin/activate
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PIP_NO_BUILD_ISOLATION=1

python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1" wheel packaging
python -m pip install "numpy==1.26.4"

# If torch is not already installed in the venv, install the CUDA 12.4 wheels first:
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install TRELLIS.2 basic Python dependencies manually so no sudo apt is required.
python -m pip install "numpy==1.26.4" imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard kornia timm
python -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install flash-attn without build isolation so its build can see torch in this venv.
python -m pip install flash-attn==2.7.3 --no-build-isolation

# Install compiled TRELLIS.2 dependencies into the active venv. Do not pass --new-env, --basic, or --flash-attn.
. ./setup.sh --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

If your GPU does not support `flash-attn`, install/use `xformers` instead and set:

```bash
export ATTN_BACKEND=xformers
```

## Smoke test

After setup:

```bash
cd third_party/TRELLIS.2
source ../../venv/bin/activate
python example.py
```

TRELLIS.2 should generate a preview video and a GLB asset from the bundled example image.

## Minimal image-to-GLB script

Create `run_trellis2_image.py` somewhere on the cluster:

```python
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
import o_voxel


def main(image_path: str, output_glb: str) -> None:
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(
            cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        ),
        dtype=torch.float32,
        device="cuda",
    ))

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    image = Image.open(image_path)
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(output_glb, extension_webp=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

Run it from inside the TRELLIS.2 checkout:

```bash
cd third_party/TRELLIS.2
source ../../venv/bin/activate
python /path/to/run_trellis2_image.py /path/to/input.png /path/to/output.glb
```

## Notes

- The first run downloads the model weights from Hugging Face, so use a node/session with network access or pre-populate the Hugging Face cache.
- Set `HF_HOME` or `HUGGINGFACE_HUB_CACHE` if your cluster has a shared model cache.
- If multiple CUDA versions are installed, set `CUDA_HOME` before running setup.
- TRELLIS.2 upstream recommends conda, but their installer supports existing environments by omitting `--new-env`.
- This helper intentionally skips upstream `--basic` because that path tries `sudo apt install -y libjpeg-dev`, which normal cluster users cannot run.
- It pins 
umpy<2` before and after installs because `utils3d` can otherwise pull NumPy 2.x back into the venv, which breaks PyTorch/CUDA extension metadata generation.
- Compiled extension sources are cached under `${TMPDIR:-/tmp}/trellis2_extensions_$USER` by default. Set `TRELLIS2_EXT_DIR=/path/to/cache` if you want that cache elsewhere.
- For Slurm, request a GPU node before running setup or inference.