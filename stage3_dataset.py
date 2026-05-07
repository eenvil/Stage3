from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXT_KEYS = {
    "reference": ["reference", "reference_path", "original", "original_path", "input", "image", "image_path"],
    "composite": ["composite_optimized", "optimized_composite", "composite", "composite_path"],
    "render": ["render_optimized", "render_optimize", "optimized_render", "render", "render_path"],
    "albedo": ["albedo", "mvinverse_albedo"],
    "normal": ["normal", "mvinverse_normal"],
    "roughness": ["roughness", "mvinverse_roughness"],
    "metallic": ["metallic", "mvinverse_metallic"],
    "shading": ["shading", "mvinverse_shading"],
}


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def resolve_path(value: Any, dataset_root: Optional[str | Path] = None) -> Optional[Path]:
    if value is None or value == "":
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    if dataset_root is not None:
        return Path(dataset_root) / p
    return p


def first_existing_path(
    row: Dict[str, Any],
    logical_name: str,
    dataset_root: Optional[str | Path] = None,
    require_exists: bool = False,
) -> Optional[Path]:
    keys = IMAGE_EXT_KEYS.get(logical_name, [logical_name])
    for key in keys:
        value = row.get(key)
        if not value:
            continue
        p = resolve_path(value, dataset_root)
        if p is None:
            continue
        if p.exists() or not require_exists:
            return p
    return None


def load_rgb(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    """
    Returns float tensor [3, H, W] in 0..1.
    size is (height, width).
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((size[1], size[0]), Image.BICUBIC)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        data = data.view(size[0], size[1], 3).permute(2, 0, 1).float() / 255.0
    return data


def load_gray(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    """
    Returns float tensor [1, H, W] in 0..1.
    """
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize((size[1], size[0]), Image.BICUBIC)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        data = data.view(size[0], size[1], 1).permute(2, 0, 1).float() / 255.0
    return data


class Stage3RelightDataset(Dataset):
    """
    Dataset for Stage3 CG-to-real / renderer-gap refinement.

    Expected JSONL row fields:
      reference
      composite_optimized
      render_optimized
      optional: albedo / mvinverse_albedo
      optional: normal / roughness / metallic / shading

    Recommended first baseline after removing mask:
      input_names=("composite", "render", "albedo")
      target_name="reference"

    Important:
      No mask is used. The previous render_nonzero mask idea is removed because
      render_optimized can contain HDR/background tile artifacts, making that mask invalid.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        dataset_root: Optional[str | Path] = None,
        image_size: int | Tuple[int, int] = 256,
        input_names: Sequence[str] = ("composite", "render", "albedo"),
        target_name: str = "reference",
        missing_optional: str = "zeros",
        require_all_inputs: bool = False,
        limit: int = -1,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        if isinstance(image_size, int):
            self.size = (image_size, image_size)
        else:
            self.size = (int(image_size[0]), int(image_size[1]))

        self.input_names = tuple(input_names)
        self.target_name = target_name
        self.missing_optional = missing_optional
        self.require_all_inputs = require_all_inputs

        rows = read_jsonl(self.jsonl_path)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)
        if limit > 0:
            rows = rows[:limit]

        self.rows: List[Dict[str, Any]] = []
        skipped = 0
        for row in rows:
            if self._row_is_usable(row):
                self.rows.append(row)
            else:
                skipped += 1

        if not self.rows:
            raise RuntimeError(f"No usable rows found in {self.jsonl_path}")

        self.skipped = skipped
        self.channels = self._compute_channels()

    def _row_is_usable(self, row: Dict[str, Any]) -> bool:
        # Required for Stage3 training.
        for name in ["reference", "composite", "render"]:
            p = first_existing_path(row, name, self.dataset_root, require_exists=True)
            if p is None:
                return False

        if self.require_all_inputs:
            for name in self.input_names:
                p = first_existing_path(row, name, self.dataset_root, require_exists=True)
                if p is None:
                    return False

        return True

    def _compute_channels(self) -> int:
        channels = 0
        for name in self.input_names:
            if name in {"roughness", "metallic"}:
                channels += 1
            else:
                channels += 3
        return channels

    def __len__(self) -> int:
        return len(self.rows)

    def _load_optional_rgb(self, row: Dict[str, Any], name: str) -> torch.Tensor:
        p = first_existing_path(row, name, self.dataset_root, require_exists=True)
        if p is not None:
            return load_rgb(p, self.size)
        if self.missing_optional == "zeros":
            return torch.zeros(3, self.size[0], self.size[1], dtype=torch.float32)
        raise FileNotFoundError(f"Missing optional RGB input {name}")

    def _load_optional_gray(self, row: Dict[str, Any], name: str) -> torch.Tensor:
        p = first_existing_path(row, name, self.dataset_root, require_exists=True)
        if p is not None:
            return load_gray(p, self.size)
        if self.missing_optional == "zeros":
            return torch.zeros(1, self.size[0], self.size[1], dtype=torch.float32)
        raise FileNotFoundError(f"Missing optional gray input {name}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        ref_path = first_existing_path(row, "reference", self.dataset_root, require_exists=True)
        comp_path = first_existing_path(row, "composite", self.dataset_root, require_exists=True)
        render_path = first_existing_path(row, "render", self.dataset_root, require_exists=True)

        assert ref_path is not None
        assert comp_path is not None
        assert render_path is not None

        reference = load_rgb(ref_path, self.size)
        composite = load_rgb(comp_path, self.size)
        render = load_rgb(render_path, self.size)

        tensors: Dict[str, torch.Tensor] = {
            "reference": reference,
            "composite": composite,
            "render": render,
        }

        for name in ["albedo", "normal", "shading"]:
            if name in self.input_names:
                tensors[name] = self._load_optional_rgb(row, name)

        for name in ["roughness", "metallic"]:
            if name in self.input_names:
                tensors[name] = self._load_optional_gray(row, name)

        inputs = []
        for name in self.input_names:
            if name not in tensors:
                if name in {"roughness", "metallic"}:
                    tensors[name] = self._load_optional_gray(row, name)
                else:
                    tensors[name] = self._load_optional_rgb(row, name)
            inputs.append(tensors[name])

        x = torch.cat(inputs, dim=0)
        y = tensors[self.target_name]

        scene_id = row.get("scene_id") or row.get("source_stem") or Path(str(ref_path)).stem

        return {
            "x": x,
            "y": y,
            "reference": reference,
            "composite": composite,
            "render": render,
            "scene_id": str(scene_id),
            "row": row,
            "paths": {
                "reference": str(ref_path),
                "composite": str(comp_path),
                "render": str(render_path),
            },
        }


def make_stage3_dataset(
    jsonl_path: str | Path,
    dataset_root: Optional[str | Path] = None,
    image_size: int = 256,
    with_albedo: bool = True,
    limit: int = -1,
) -> Stage3RelightDataset:
    inputs = ["composite", "render"]
    if with_albedo:
        inputs.append("albedo")

    return Stage3RelightDataset(
        jsonl_path=jsonl_path,
        dataset_root=dataset_root,
        image_size=image_size,
        input_names=inputs,
        target_name="reference",
        missing_optional="zeros",
        require_all_inputs=False,
        limit=limit,
    )
