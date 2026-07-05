# ImageToGarment Setup

This workspace contains the two repositories needed to run ChatGarment:

```text
ImageToGarment/
  ChatGarment/
  GarmentCodeRC/
  venv/
```

Use the cluster setup guide here:

```text
ChatGarment/CLUSTER_SETUP.md
```

That guide documents the working installation path for this workspace:

```text
Python 3.10
CUDA toolkit 11.8
torch==2.1.2+cu118
torchvision==0.16.2+cu118
flash-attn==2.5.8 built from source
setuptools==69.5.1
```

## Quick Start

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
source venv/bin/activate
cd ChatGarment
```

Verify the core packages:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flash_attn; print('flash-attn ok')"
python -c "import llava; print('llava ok')"
python -c "import pygarment; print('pygarment ok')"
```

The checkpoint should be located at:

```text
ChatGarment/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin
```

For full installation, troubleshooting, checkpoint notes, and path fixes, read:

```text
ChatGarment/CLUSTER_SETUP.md
```
