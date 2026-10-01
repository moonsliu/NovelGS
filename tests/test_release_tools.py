from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from common import parse_indices


def collect_targets(value):
    if isinstance(value, dict):
        if isinstance(value.get("target"), str):
            yield value["target"]
        for child in value.values():
            yield from collect_targets(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_targets(child)


class ParseIndicesTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_indices("17,13,18,28", expected=4), [17, 13, 18, 28])

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_indices("1,1")

    def test_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 4"):
            parse_indices("1,2", expected=4)


class InputValidationTests(unittest.TestCase):
    def test_missing_camera_archive(self):
        from common import load_posed_inputs
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "cameras.npz"):
                load_posed_inputs(directory, [0, 1, 2, 3], 4, 512, 50.0)


class TrainingConfigTests(unittest.TestCase):
    def test_all_config_targets_are_importable(self):
        from omegaconf import OmegaConf
        from src.utils.train_util import get_obj_from_str

        root = Path(__file__).resolve().parents[1]
        for name in ("train_256.yaml", "train_512.yaml"):
            config = OmegaConf.to_container(
                OmegaConf.load(root / "configs" / name), resolve=True
            )
            targets = sorted(set(collect_targets(config)))
            self.assertTrue(targets, name)
            for target in targets:
                with self.subTest(config=name, target=target):
                    self.assertTrue(callable(get_obj_from_str(target)))


if __name__ == "__main__":
    unittest.main()
