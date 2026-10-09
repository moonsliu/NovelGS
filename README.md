# NovelGS

Official release for **NovelGS: Consistent Novel-view Denoising via Large Gaussian Reconstruction Model**.

- Paper: https://arxiv.org/abs/2411.16779
- Code: https://github.com/moonsliu/NovelGS
- Model weights: https://huggingface.co/moonsliu/NovelGS

## What is included

NovelGS reconstructs a 3D Gaussian representation from four posed input views and performs iterative novel-view denoising at a target camera. This release contains the paper model, two-stage training configuration, posed multi-view inference, an optional Zero123++ single-image pipeline, and evaluation utilities.

The training data, third-party model weights, experiment logs, and historical outputs are not distributed.

## Installation

Linux, an NVIDIA GPU, and CUDA are required. The tested release environment targets Python 3.10 and CUDA 12.8.

```bash
git clone --recurse-submodules https://github.com/moonsliu/NovelGS.git
cd NovelGS
conda env create -f environment.yml
conda activate novelgs
pip install --no-build-isolation -v ./third_party/diff-gaussian-rasterization
```

For an existing clone, initialize the rasterizer with `git submodule update --init --recursive`.

The CUDA extension in `third_party/diff-gaussian-rasterization` is a modified research-only component. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before use.

## Weights

Download `model.safetensors` from the Hugging Face repository. It contains the 353 sanitized NovelGS tensors without optimizer, trainer, callback, or frozen LPIPS/VGG state.

```bash
huggingface-cli download moonsliu/NovelGS model.safetensors \
  --local-dir checkpoints
```

## Posed multi-view inference

Input directory layout:

```text
object/
├── 000.png
├── 001.png
├── ...
└── cameras.npz       # cam_poses: [N, 3, 4], OpenGL world-to-camera
```

Images must be RGBA. Four condition indices and one or more target-camera indices are required. Multiple targets are processed independently.

```bash
python tools/infer_multiview.py \
  --input-dir path/to/object \
  --checkpoint checkpoints/model.safetensors \
  --condition-indices 17,13,18,28 \
  --target-indices 26 \
  --output-dir outputs/example \
  --save-video \
  --save-mesh
```

Each target produces a denoised target image and Gaussian PLY; `--save-video` writes an orbit MP4 and `--save-mesh` writes an approximate Poisson mesh.

## Optional single-image inference

Zero123++ is downloaded at runtime and its weights are licensed CC-BY-NC 4.0. Therefore this optional pipeline is non-commercial even though NovelGS's own code and weights are Apache-2.0.

```bash
pip install -r requirements-single-image.txt
python tools/infer_single_image.py \
  --image path/to/input.png \
  --checkpoint checkpoints/model.safetensors \
  --output-dir outputs/single \
  --save-video \
  --save-mesh
```

No Zero123++, InstantMesh, SV3D, or Stable Diffusion weights are bundled.

## Training

Raw Objaverse assets are not directly consumable. Prepare 32 RGBA renders and camera matrices as described in [docs/dataset.md](docs/dataset.md).

```bash
python train.py --base configs/train_256.yaml \
  --data_root /path/to/prepared_objaverse \
  --split_file splits/train.json \
  --gpus 0,1,2,3

python train.py --base configs/train_512.yaml \
  --data_root /path/to/prepared_objaverse \
  --split_file splits/train.json \
  --resume /path/to/stage1.ckpt --resume_weights_only \
  --gpus 0,1,2,3
```

Additional OmegaConf overrides may be appended, for example `lightning.trainer.max_steps=10`.

## Evaluation

```bash
python tools/evaluate.py --manifest configs/eval/example_manifest.json \
  --output outputs/metrics.json
```

Evaluation manifests pair prediction and ground-truth image files. Copy `configs/eval/example_manifest.json` and replace its paths with locally prepared predictions and RGBA targets.

## Citation

```bibtex
@article{liu2024novelgs,
  title={NovelGS: Consistent Novel-view Denoising via Large Gaussian Reconstruction Model},
  author={Liu, Jinpeng and Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Shan, Ying and Tang, Yansong},
  journal={arXiv preprint arXiv:2411.16779},
  year={2024}
}
```

## License

NovelGS-authored code and released NovelGS weights are Apache-2.0. Subdirectories and optional models may have more restrictive licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
