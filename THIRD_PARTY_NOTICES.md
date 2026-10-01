# Third-party notices

NovelGS-authored code and NovelGS model weights are licensed under Apache License 2.0. That license does not replace licenses of third-party components, datasets, or optional weights.

## diff-gaussian-rasterization

The Git submodule under `third_party/diff-gaussian-rasterization` points to the NovelGS fork of the GraphDECO Gaussian Splatting rasterizer and retains its original `LICENSE.md`. It is restricted to non-commercial research/evaluation use under the terms stated there.

This fork modifies CUDA forward/backward paths to expose depth/alpha values and gradients required by NovelGS. Compiled libraries and build artifacts are intentionally excluded.


## GLM

The GLM submodule nested under the rasterizer retains the upstream Modified MIT/MIT terms reproduced in `third_party/diff-gaussian-rasterization/third_party/glm/manual.md`.

## Zero123++

Optional single-image view generation uses https://github.com/SUDO-AI-3D/zero123plus.

- code: Apache-2.0;
- model weights: CC-BY-NC 4.0.

The weights are downloaded from their official Hugging Face repository and are not distributed here. Using the optional single-image pipeline introduces the non-commercial weight restriction.

## InstantMesh

The optional white-background Zero123++ UNet is downloaded from https://github.com/TencentARC/InstantMesh / `TencentARC/InstantMesh`. It is not distributed here. Review the repository and model-card terms before use.

## Objaverse

Objaverse is not redistributed. Dataset-level and per-object licenses apply independently; consult https://objaverse.allenai.org/ and the metadata returned by the official API.

## Other packages

PyTorch, Lightning, diffusers, transformers, xformers, Open3D, rembg, LPIPS, and other Python dependencies are installed from their upstream distributions. Their licenses remain unchanged. No LPIPS/VGG, SV3D, Stable Diffusion 3.5, or other third-party checkpoints are bundled.
