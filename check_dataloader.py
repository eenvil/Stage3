from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from stage3_dataset import Stage3RelightDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--no-albedo", action="store_true")
    args = ap.parse_args()

    inputs = ["composite", "render"]
    if not args.no_albedo:
        inputs.append("albedo")

    ds = Stage3RelightDataset(
        jsonl_path=args.jsonl,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        input_names=inputs,
        target_name="reference",
        missing_optional="zeros",
        require_all_inputs=False,
        limit=args.limit,
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(dl))

    print("[INFO] dataset rows:", len(ds))
    print("[INFO] skipped rows :", ds.skipped)
    print("[INFO] input names  :", inputs)
    print("[INFO] channels     :", ds.channels)
    print("[BATCH] x        :", tuple(batch["x"].shape), batch["x"].dtype, float(batch["x"].min()), float(batch["x"].max()))
    print("[BATCH] y        :", tuple(batch["y"].shape), batch["y"].dtype, float(batch["y"].min()), float(batch["y"].max()))
    print("[BATCH] composite:", tuple(batch["composite"].shape))
    print("[BATCH] render   :", tuple(batch["render"].shape))
    print("[BATCH] scene_id :", batch["scene_id"][: min(4, len(batch["scene_id"]))])


if __name__ == "__main__":
    main()
