#!/usr/bin/env python3
"""Optional Zero123++ to NovelGS single-image pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_512.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=75)
    parser.add_argument("--novelgs-steps", type=int, default=75)
    parser.add_argument("--zero123-model", default="sudo-ai/zero123plus-v1.2")
    parser.add_argument("--zero123-pipeline", default="sudo-ai/zero123plus-pipeline")
    parser.add_argument("--instantmesh-repo", default="TencentARC/InstantMesh")
    parser.add_argument("--instantmesh-unet", default="diffusion_pytorch_model.bin")
    parser.add_argument("--skip-instantmesh-unet", action="store_true")
    parser.add_argument("--no-remove-background", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-mesh", action="store_true", help="export an approximate Open3D Poisson mesh")
    parser.add_argument("--mesh-depth", type=int, default=8, help="Poisson depth in [5,12]")
    parser.add_argument("--orbit-views", type=int, default=120)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.diffusion_steps <= 0:
        raise ValueError("--diffusion-steps must be positive")
    if args.novelgs_steps <= 0:
        raise ValueError("--novelgs-steps must be positive")
    if args.orbit_views <= 0:
        raise ValueError("--orbit-views must be positive")
    if args.save_mesh and not 5 <= args.mesh_depth <= 12:
        raise ValueError("--mesh-depth must be between 5 and 12")
    try:
        import cv2
        import numpy as np
        import rembg
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        from einops import rearrange
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "single-image dependencies are missing; run "
            "`pip install -r requirements-single-image.txt`"
        ) from exc

    from common import circular_cameras, load_model, save_gaussians, save_mesh, save_tensor_image, save_video, write_json
    from src.utils.camera_util import get_zero123plus_input_cameras, pad_image_to_fit_fov
    from src.utils.infer_util import remove_background, resize_foreground

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")
    source = Path(args.image)
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipeline = DiffusionPipeline.from_pretrained(
        args.zero123_model, custom_pipeline=args.zero123_pipeline, torch_dtype=dtype
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    if not args.skip_instantmesh_unet:
        unet_path = hf_hub_download(
            repo_id=args.instantmesh_repo, filename=args.instantmesh_unet, repo_type="model"
        )
        payload = torch.load(unet_path, map_location="cpu", weights_only=True)
        pipeline.unet.load_state_dict(payload, strict=True)
    pipeline.to(args.device)

    image = Image.open(source).convert("RGBA")
    session = None
    if not args.no_remove_background:
        session = rembg.new_session()
        image = resize_foreground(remove_background(image, session), 0.85)
    grid = pipeline(image, num_inference_steps=args.diffusion_steps).images[0]
    grid.save(output / "zero123plus_grid.png")
    generated = torch.from_numpy(np.asarray(grid.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    generated = rearrange(generated, "c (rows h) (cols w) -> (rows cols) c h w", rows=3, cols=2)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = OmegaConf.load(args.config)
    model_cfg = config.get("model_config", config.get("model"))
    size = int(model_cfg.params.image_size)
    prepared = []
    for view in generated:
        array = (view.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        array = pad_image_to_fit_fov(array, 50, 30)
        if args.no_remove_background:
            rgb = np.asarray(Image.fromarray(array).convert("RGB"), dtype=np.float32) / 255.0
        else:
            rgba = remove_background(Image.fromarray(array), session)
            rgba = np.asarray(rgba, dtype=np.float32) / 255.0
            alpha = rgba[..., 3:4]
            rgb = rgba[..., :3] * alpha + (1.0 - alpha)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        prepared.append(torch.from_numpy(rgb).permute(2, 0, 1).float())
    views = (torch.stack(prepared)[[0, 2, 4, 5]].unsqueeze(0) - 0.5) * 2.0

    cameras, transform = get_zero123plus_input_cameras(batch_size=1, radius=2.0, fov=50.0)
    camera_order = torch.tensor([0, 2, 4, 5, 1])
    c2ws = cameras[:, camera_order, :16].reshape(1, 5, 4, 4)
    intrinsics = cameras[:, 0, 16:].unsqueeze(1).repeat(1, 5, 1)
    orbit_c2ws, orbit_intrinsics = circular_cameras(args.orbit_views, 1.5, 0.0, 50.0)
    orbit_c2ws = transform.unsqueeze(0) @ orbit_c2ws

    model = load_model(args.config, args.checkpoint, args.device, args.novelgs_steps)
    with torch.inference_mode():
        samples, frames, gaussians = model.val4to9(
            views.to(args.device), c2ws.to(args.device), intrinsics.to(args.device),
            orbit_c2ws.to(args.device), orbit_intrinsics.to(args.device),
        )
    save_tensor_image(samples[0, 0], output / "denoised.png")
    save_gaussians(model, gaussians, output / "gaussians.ply", args.opacity_threshold)
    if args.save_mesh:
        save_mesh(gaussians, output / "mesh.ply", args.opacity_threshold, args.mesh_depth)
    if args.save_video:
        save_video(frames, output / "orbit.mp4")
    write_json(
        {
            "input": str(source.resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "zero123_model": args.zero123_model,
            "zero123_weights_license": "CC-BY-NC-4.0",
            "seed": args.seed,
            "mesh_depth": args.mesh_depth if args.save_mesh else None,
        },
        output / "run.json",
    )
    print(f"saved results to {output}")


if __name__ == "__main__":
    main()
