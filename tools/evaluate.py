#!/usr/bin/env python3
"""Compute PSNR, SSIM, and LPIPS for prediction/target image pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON object with a 'pairs' list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from lpips import LPIPS
    from skimage.metrics import structural_similarity

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("manifest must contain a non-empty 'pairs' list")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    metric = LPIPS(net="alex").eval().to(args.device)

    def load(path: str, target: bool):
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(file)
        image = Image.open(file)
        if target:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
        image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if target:
            alpha = array[..., 3:4]
            return array[..., :3] * alpha + (1.0 - alpha), alpha
        return array[..., :3], None

    rows = []
    with torch.inference_mode():
        for item in pairs:
            pred, _ = load(item["prediction"], target=False)
            target, mask = load(item["target"], target=True)
            mse = float(np.mean(((pred - target) ** 2) * mask))
            psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
            ssim = float(structural_similarity(pred, target, data_range=1.0, channel_axis=-1))
            pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(args.device)
            target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(args.device)
            lpips_value = float(metric(pred_t, target_t, normalize=True).item())
            rows.append({"prediction": item["prediction"], "target": item["target"], "mse": mse, "psnr": psnr, "ssim": ssim, "lpips": lpips_value})

    summary = {name: float(np.mean([row[name] for row in rows])) for name in ("mse", "psnr", "ssim", "lpips")}
    result = {"dataset": manifest.get("dataset", "unspecified"), "count": len(rows), "summary": summary, "pairs": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
