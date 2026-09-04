# Third-party components

This repository contains the v3b orchestration and project-specific processing
code. It intentionally does not redistribute datasets, external repositories,
or model weights. Install every external component under its own upstream
license, preferably in a separate Python environment, and put only local paths
in `config/v3b.env`.

## Upstream repositories

| Component | Upstream | Revision used for the three-room regression |
|---|---|---|
| Depth Anything 3 | <https://github.com/ByteDance-Seed/Depth-Anything-3> | `41736238f5bced4debf3f2a12375d2466874866d` |
| RoomFormer | <https://github.com/ywyue/RoomFormer> | `e88a7e3a81e384e15ea5bdc02d893267a2b6cac1` |
| SAM 3 | <https://github.com/facebookresearch/sam3> | `2814fa619404a722d03e9a012e083e4f293a4e53` |
| MatSeg zero-shot | <https://github.com/sagieppel/Zero-Shot-Material-State-Segmentation-Net> | `3e6be932fde242a881fc599a73f3f61c884b4764` |
| Ubisoft LaForge CHORD | <https://github.com/ubisoft/ubisoft-laforge-chord> | `675a29d1fee5e1b256833bd4edabd619be43d0c7` |
| IOPaint/LaMa | <https://github.com/Sanster/IOPaint> | IOPaint `1.5.3` |

The tested DA3 weights are the Hugging Face snapshot
`depth-anything/DA3-LARGE-1.1@0e109ae307c5982f319a67cf6f9f99ccdc0ec97c`.
Set `DA3_MODEL_DIR` to the downloaded snapshot directory. Follow each upstream
project’s installation instructions for its own runtime and weights.

For IOPaint 1.5.3, `--model-dir` and PyTorch's model cache are separate. Set
both `IOPAINT_MODEL_DIR` and `TORCH_HOME`; Big-LaMa is expected below
`$TORCH_HOME/hub/checkpoints/big-lama.pt`. If it is absent and networking is
available, IOPaint downloads it on first use.

## Tested checkpoint fingerprints

The filenames themselves are not sufficient proof that the same weights were
used. The accepted regression used:

```text
roomformer_stru3d_tight.pth
  sha256 94d82eacfc37ea19d786ddbb0f8b507eabfb0d702ff892033717f9bfcbc5c4cf
roomformer_stru3d.pth
  sha256 1bd96a508ac0cc3d1b2a83eb77ff732014fc7336f0a904a12f0de884aef91247
Defult.torch
  sha256 4268fe4fa6064c801f8333426c61da5c1d557dd303189ab833030b8a4b1d63b6
chord_v1.safetensors
  sha256 b56f48857f1d84f0d051284d0c1f3ac58ecc6e86fba15c4849789f3807dd0cc2
```

## Environment split

The verified installation used Python 3.10 for DA3, RoomFormer, the main
reconstruction/MatSeg environment, and CHORD; SAM 3 used Python 3.12. The main
environment had PyTorch 2.5.1, torchvision 0.20.1, OpenCV 4.11, NumPy 1.26.4,
SciPy 1.11.4, and IOPaint 1.5.3. CHORD used PyTorch 2.5.1+cu121, diffusers
0.38.0, and transformers 4.57.1.

`requirements-runtime.txt` covers the ordinary geometry/image libraries used
by this repository. It does not replace the installation requirements of DA3,
RoomFormer, SAM 3, MatSeg, CHORD, or IOPaint.

Before presenting this repository as open-source software, add a project
license chosen by the repository owner. No project license is asserted here.
