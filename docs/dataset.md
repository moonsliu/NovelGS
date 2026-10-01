# Dataset preparation

NovelGS was trained from rendered Objaverse 1.0 objects. The original dataset and the authors' rendered training corpus are not redistributed.

## Download source objects

Use the official Objaverse project and API:

- https://objaverse.allenai.org/objaverse-1.0/
- https://github.com/allenai/objaverse-xl
- https://huggingface.co/datasets/allenai/objaverse

Each source object retains its own license. Check metadata before downloading, rendering, redistributing, or using an object.

## Required prepared layout

The training loader expects:

```text
DATA_ROOT/
├── rendering_random_32views/
│   └── OBJECT_UID/
│       ├── 000.png
│       ├── 001.png
│       ├── ...
│       ├── 031.png
│       └── cameras.npz
└── splits/
    └── train.json
```

Requirements:

- exactly 32 views per object, numbered `000.png` through `031.png`;
- 512×512 RGBA PNG files with transparency representing the object mask;
- a nominal 50-degree field of view and centered principal point;
- `cameras.npz` containing `cam_poses` with shape `[32, 3, 4]`;
- `cam_poses` are OpenGL world-to-camera matrices;
- `train.json` is a JSON list of object UID strings.

The 256-pixel stage resizes the same renderings. The 512-pixel stage consumes them at native resolution.

## Important limitation

Downloading raw Objaverse objects is not sufficient to run training. A Blender-based rendering/filtering process must produce the layout above. The exact historical rendering farm and the filtered 270K-object split were not recoverable, so this release documents the contract but does not claim bit-exact dataset reproduction.
