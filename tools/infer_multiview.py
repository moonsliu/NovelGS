#!/usr/bin/env python3
"""Run NovelGS on four posed RGBA views and one or more target cameras."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="directory containing NNN.png and cameras.npz")
    parser.add_argument("--checkpoint", required=True, help="trusted Lightning .ckpt or model-only .safetensors")
    parser.add_argument("--config", default="configs/train_512.yaml")
    parser.add_argument("--condition-indices", default="17,13,18,28", help="comma-separated indices; four for the paper checkpoint")
    parser.add_argument("--target-indices", required=True, help="comma-separated target camera indices")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=75)
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument("--orbit-views", type=int, default=120)
    parser.add_argument("--orbit-radius", type=float, default=2.5)
    parser.add_argument("--orbit-elevation", type=float, default=0.0)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-mesh", action="store_true", help="export an approximate Open3D Poisson mesh")
    parser.add_argument("--mesh-depth", type=int, default=8, help="Poisson depth in [5,12]")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from omegaconf import OmegaConf
    from common import (
        circular_cameras, load_model, load_posed_inputs, parse_indices,
        save_gaussians, save_mesh, save_tensor_image, save_video, write_json,
    )

    if args.inference_steps <= 0:
        raise ValueError("--inference-steps must be positive")
    if args.orbit_views <= 0:
        raise ValueError("--orbit-views must be positive")
    if args.save_mesh and not 5 <= args.mesh_depth <= 12:
        raise ValueError("--mesh-depth must be between 5 and 12")
    condition = parse_indices(args.condition_indices)
    targets = parse_indices(args.target_indices)
    overlap = sorted(set(condition) & set(targets))
    if overlap:
        raise ValueError(f"condition and target indices overlap: {overlap}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    config = OmegaConf.load(args.config)
    model_cfg = config.get("model_config", config.get("model"))
    image_size = int(model_cfg.params.image_size)
    model = load_model(args.config, args.checkpoint, args.device, args.inference_steps)
    output_root = Path(args.output_dir)

    for target in targets:
        images, c2ws, intrinsics = load_posed_inputs(
            args.input_dir, condition, target, image_size=image_size, fov=args.fov
        )
        orbit_c2ws, orbit_intrinsics = circular_cameras(
            args.orbit_views, args.orbit_radius, args.orbit_elevation, args.fov
        )
        images = images.to(args.device)
        c2ws = c2ws.to(args.device)
        intrinsics = intrinsics.to(args.device)
        orbit_c2ws = orbit_c2ws.to(args.device)
        orbit_intrinsics = orbit_intrinsics.to(args.device)

        with torch.inference_mode():
            samples, frames, gaussians = model.val4to9(
                images, c2ws, intrinsics, orbit_c2ws, orbit_intrinsics
            )

        target_dir = output_root / f"target_{target:03d}"
        save_tensor_image(samples[0, 0], target_dir / "denoised.png")
        save_gaussians(model, gaussians, target_dir / "gaussians.ply", args.opacity_threshold)
        if args.save_mesh:
            save_mesh(gaussians, target_dir / "mesh.ply", args.opacity_threshold, args.mesh_depth)
        if args.save_video:
            save_video(frames, target_dir / "orbit.mp4")
        write_json(
            {
                "condition_indices": condition,
                "target_index": target,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "config": str(Path(args.config).resolve()),
                "inference_steps": args.inference_steps,
                "seed": args.seed,
                "fov": args.fov,
                "mesh_depth": args.mesh_depth if args.save_mesh else None,
            },
            target_dir / "run.json",
        )
        print(f"saved target {target} to {target_dir}")


if __name__ == "__main__":
    main()
