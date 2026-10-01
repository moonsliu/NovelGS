"""Shared release utilities for NovelGS command-line tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def parse_indices(value: str, expected: int | None = None) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"indices must be comma-separated integers: {value!r}") from exc
    if not result:
        raise ValueError("at least one index is required")
    if len(set(result)) != len(result):
        raise ValueError(f"indices must be unique: {result}")
    if expected is not None and len(result) != expected:
        raise ValueError(f"expected {expected} indices, got {len(result)}")
    if min(result) < 0:
        raise ValueError("indices must be non-negative")
    return result


def load_model(config_path: str, checkpoint_path: str, device: str, inference_steps: int | None = None):
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from src.utils.train_util import instantiate_from_config

    config = OmegaConf.load(config_path)
    model_config = config.get("model_config", config.get("model"))
    if model_config is None:
        raise ValueError(f"{config_path} contains neither 'model' nor 'model_config'")
    model = instantiate_from_config(model_config)

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if checkpoint.name.endswith(".safetensors.index.json"):
        index = json.loads(checkpoint.read_text(encoding="utf-8"))
        state_dict = {}
        for filename in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(checkpoint.parent / filename), device="cpu"))
    elif checkpoint.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint), device="cpu")
    else:
        # Lightning checkpoints use pickle. Only load files obtained from a trusted source.
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=False)
    required_missing = [key for key in incompatible.missing_keys if not key.startswith("lpips.")]
    if required_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch:\n"
            f"missing={required_missing[:20]}\n"
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    if inference_steps is not None:
        model.num_inference_steps = inference_steps
    model.eval().to(torch.device(device))
    return model


def _load_rgba(path: Path, size: int):
    import numpy as np
    import torch
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    image = Image.open(path)
    if image.mode != "RGBA":
        raise ValueError(f"expected an RGBA image, got {image.mode}: {path}")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    rgba = np.asarray(image, dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() * 2.0 - 1.0


def load_posed_inputs(
    input_dir: str,
    condition_indices: Iterable[int],
    target_index: int,
    image_size: int,
    fov: float,
):
    import numpy as np
    import torch

    root = Path(input_dir)
    camera_path = root / "cameras.npz"
    if not camera_path.is_file():
        raise FileNotFoundError(f"missing camera archive: {camera_path}")
    archive = np.load(camera_path)
    if "cam_poses" not in archive:
        raise KeyError(f"{camera_path} must contain 'cam_poses'")
    poses = archive["cam_poses"]
    if poses.ndim != 3 or poses.shape[1:] != (3, 4):
        raise ValueError(f"cam_poses must have shape [N,3,4], got {poses.shape}")

    condition_indices = list(condition_indices)
    selected = condition_indices + [target_index]
    if max(selected) >= len(poses):
        raise IndexError(f"camera index {max(selected)} is outside N={len(poses)}")

    images = torch.stack([_load_rgba(root / f"{idx:03d}.png", image_size) for idx in condition_indices])
    selected_poses = poses[selected]
    bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=selected_poses.dtype), (len(selected), 1, 4))
    w2cs = torch.from_numpy(np.concatenate([selected_poses, bottom], axis=1)).float()
    c2ws = torch.linalg.inv(w2cs)

    canonical_distance = torch.tensor(2.0)
    scale = canonical_distance / c2ws[0, :3, 3].norm()
    c2ws[:, :3, 3] *= scale
    from src.utils.camera_util import create_blender_camera
    canonical = create_blender_camera(torch.tensor([[0.0, -2.0, 0.0]])).to(c2ws)
    transform = canonical @ torch.linalg.inv(c2ws[0:1])
    c2ws = transform @ c2ws
    c2ws[:, :3, 1:3] *= -1  # OpenGL to OpenCV convention used by the model.

    focal = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([focal, focal, 0.5, 0.5], dtype=torch.float32).repeat(len(selected), 1)
    return images.unsqueeze(0), c2ws.unsqueeze(0), intrinsics.unsqueeze(0)


def circular_cameras(count: int, radius: float, elevation: float, fov: float):
    import numpy as np
    import torch
    from src.utils.camera_util import get_circular_camera_poses

    c2ws = get_circular_camera_poses(M=count, radius=radius, elevation=elevation)
    c2ws[:, :3, 1:3] *= -1
    focal = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([focal, focal, 0.5, 0.5], dtype=torch.float32).repeat(count, 1)
    return c2ws.unsqueeze(0), intrinsics.unsqueeze(0)


def save_tensor_image(tensor, path: Path) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.detach().float().cpu().clamp(-1, 1), str(path), normalize=True, value_range=(-1, 1))


def save_video(frames, path: Path, fps: int = 30) -> None:
    import imageio.v2 as imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    array = frames[0].permute(0, 2, 3, 1).detach().float().cpu().numpy()
    array = (array.clip(0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(str(path), array, fps=fps, codec="libx264")


def save_gaussians(model, latent: dict, path: Path, opacity_threshold: float = 0.1) -> None:
    import torch

    required = ("xyz", "feature", "opacity", "scaling", "rotation")
    missing = [key for key in required if key not in latent]
    if missing:
        raise KeyError(f"Gaussian output missing keys: {missing}")
    keep = torch.nonzero(latent["opacity"][0].sigmoid().squeeze(-1) > opacity_threshold).squeeze(-1)
    if keep.numel() == 0:
        raise RuntimeError("all Gaussians were removed by the opacity threshold")
    values = {key: latent[key][0, keep].to(torch.float32) for key in required}
    cloud = model.gs.gaussian_model.set_data(
        values["xyz"], values["feature"], values["scaling"], values["rotation"], values["opacity"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.save_ply(str(path))



def save_mesh(latent: dict, path: Path, opacity_threshold: float = 0.1, depth: int = 8) -> None:
    """Create an approximate triangle mesh from filtered Gaussian centers."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("mesh export requires open3d; install requirements.txt") from exc
    import numpy as np
    import torch

    if not 5 <= depth <= 12:
        raise ValueError("mesh Poisson depth must be between 5 and 12")
    required = ("xyz", "feature", "opacity")
    missing = [key for key in required if key not in latent]
    if missing:
        raise KeyError(f"Gaussian output missing keys for mesh export: {missing}")
    keep = torch.nonzero(latent["opacity"][0].sigmoid().squeeze(-1) > opacity_threshold).squeeze(-1)
    if keep.numel() < 100:
        raise RuntimeError(
            f"mesh export retained only {keep.numel()} Gaussians; "
            "lower --opacity-threshold or omit --save-mesh"
        )
    points = latent["xyz"][0, keep].detach().float().cpu().numpy()
    colors = latent["feature"][0, keep].detach().float().sigmoid().clamp(0, 1).cpu().numpy()
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError(f"expected Gaussian RGB features shaped [N,3+], got {colors.shape}")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64, copy=False))
    extent = float(np.max(np.ptp(points, axis=0)))
    radius = max(extent * 0.05, 1e-3)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    cloud.orient_normals_consistent_tangent_plane(min(20, len(points) - 1))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    if len(mesh.triangles) == 0:
        raise RuntimeError("Open3D produced an empty mesh; omit --save-mesh or lower the opacity threshold")
    mesh.compute_vertex_normals()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise RuntimeError(f"failed to write mesh: {path}")


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
