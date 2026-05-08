from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from stage3_dataset import Stage3RelightDataset
from stage3_unet import ResidualUNet


OPTIONAL_INPUT_KEYS = ["albedo", "normal", "roughness", "metallic", "shading", "glass_mask"]


def read_ids_from_file(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                s = s.split(",", 1)[0].strip()
            ids.append(s)
    return ids


def parse_scene_ids(scene_ids: Optional[str], scene_list: Optional[Path]) -> Optional[Set[str]]:
    ids: List[str] = []
    if scene_ids:
        ids.extend([x.strip() for x in scene_ids.split(",") if x.strip()])
    if scene_list is not None:
        ids.extend(read_ids_from_file(scene_list))
    return set(ids) if ids else None


def get_row_scene_id(row: Dict[str, Any]) -> str:
    for key in ["scene_id", "source_stem", "sample_id", "id"]:
        if row.get(key):
            return str(row[key])
    for key in ["reference", "reference_path", "original", "original_path", "input", "image", "image_path"]:
        if row.get(key):
            return Path(str(row[key])).stem
    return "unknown_scene"


def filter_dataset_rows(ds: Stage3RelightDataset, wanted: Optional[Set[str]], limit: int) -> None:
    if wanted is not None:
        before = len(ds.rows)
        ds.rows = [r for r in ds.rows if get_row_scene_id(r) in wanted]
        print(f"[INFO] filtered by scene ids: {before} -> {len(ds.rows)}")
    if limit > 0:
        ds.rows = ds.rows[:limit]
        print(f"[INFO] applied limit: {len(ds.rows)}")


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_rgb_tensor(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(x).save(path)


def abs_error_rgb(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1).clamp(0, 1)


def make_contact_sheet(
    path: Path,
    reference: torch.Tensor,
    composite: torch.Tensor,
    render: torch.Tensor,
    prediction: torch.Tensor,
    scene_id: str,
    optional_inputs: Dict[str, torch.Tensor],
) -> None:
    before_err = abs_error_rgb(composite, reference)
    after_err = abs_error_rgb(prediction, reference)

    cols = [
        ("reference", reference),
        ("composite_in", composite),
        ("render_in", render),
    ]

    for key in OPTIONAL_INPUT_KEYS:
        if key in optional_inputs:
            cols.append((f"{key}_in", optional_inputs[key]))

    cols.extend([
        ("prediction", prediction),
        ("err_before", before_err),
        ("err_after", after_err),
    ])

    imgs = [(label, tensor_to_pil(t)) for label, t in cols]

    w, h = imgs[0][1].size
    label_h = 24
    title_h = 28
    gap = 6
    canvas = Image.new(
        "RGB",
        (len(imgs) * w + (len(imgs) + 1) * gap, h + label_h + title_h + 3 * gap),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), scene_id, fill=(0, 0, 0))

    x = gap
    y = gap + title_h
    for label, img in imgs:
        draw.text((x, y), label, fill=(0, 0, 0))
        canvas.paste(img, (x, y + label_h))
        x += w + gap

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, ref: torch.Tensor, comp: torch.Tensor) -> Dict[str, float]:
    import math

    pred = pred.float().clamp(0, 1)
    ref = ref.float().clamp(0, 1)
    comp = comp.float().clamp(0, 1)

    pred_l1 = F.l1_loss(pred, ref).item()
    comp_l1 = F.l1_loss(comp, ref).item()
    pred_mse = F.mse_loss(pred, ref).item()
    comp_mse = F.mse_loss(comp, ref).item()

    def psnr(mse: float) -> float:
        return -10.0 * math.log10(max(mse, 1e-12))

    return {
        "composite_l1": comp_l1,
        "prediction_l1": pred_l1,
        "l1_improvement": comp_l1 - pred_l1,
        "composite_psnr": psnr(comp_mse),
        "prediction_psnr": psnr(pred_mse),
        "psnr_improvement": psnr(pred_mse) - psnr(comp_mse),
    }


def find_checkpoint(run_dir: Optional[Path], checkpoint: Optional[Path]) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    if run_dir is None:
        raise ValueError("Provide either --run-dir or --checkpoint")
    for p in [run_dir / "checkpoint_final.pt", run_dir / "checkpoint_last.pt"]:
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint_final.pt or checkpoint_last.pt found in {run_dir}")


def infer_channels_from_input_names(input_names: List[str]) -> int:
    channels = 0
    for name in input_names:
        channels += 1 if name in {"roughness", "metallic", "glass_mask"} else 3
    return channels


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    input_names = ckpt.get("input_names") or ckpt.get("inputs")
    if input_names is None:
        raise RuntimeError("Checkpoint does not contain input_names/inputs.")

    input_names = list(input_names)
    channels = int(ckpt.get("channels", 0)) or infer_channels_from_input_names(input_names)
    base_channels = int(ckpt.get("base_channels", 32))

    model = ResidualUNet(input_channels=channels, base_channels=base_channels)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, input_names, channels, base_channels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scene-ids", default=None, help="Comma-separated scene IDs")
    ap.add_argument("--scene-list", type=Path, default=None, help="Text file with one scene_id per line")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--save-inputs", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt_path = find_checkpoint(args.run_dir, args.checkpoint)
    model, input_names, channels, base_channels = load_model_from_checkpoint(ckpt_path, device)

    print("[INFO] checkpoint   :", ckpt_path)
    print("[INFO] device       :", device)
    print("[INFO] input_names  :", input_names)
    print("[INFO] channels     :", channels)
    print("[INFO] base_channels:", base_channels)

    ds = Stage3RelightDataset(
        jsonl_path=args.jsonl,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        input_names=input_names,
        target_name="reference",
        missing_optional="zeros",
        require_all_inputs=False,
        limit=-1,
        shuffle=False,
    )

    wanted = parse_scene_ids(args.scene_ids, args.scene_list)
    filter_dataset_rows(ds, wanted=wanted, limit=args.limit)

    if len(ds) == 0:
        raise RuntimeError("No scenes selected. Check --scene-ids / --scene-list and JSONL.")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: List[Dict[str, Any]] = []

    for batch in dl:
        x = batch["x"].to(device, non_blocking=True)
        comp = batch["composite"].to(device, non_blocking=True)
        pred = model(x, comp).cpu()

        for i in range(pred.shape[0]):
            scene_id = str(batch["scene_id"][i])
            scene_dir = args.out_dir / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)

            reference = batch["reference"][i]
            composite = batch["composite"][i]
            render = batch["render"][i]
            prediction = pred[i]

            save_rgb_tensor(scene_dir / "prediction.png", prediction)

            optional_inputs: Dict[str, torch.Tensor] = {}
            for key in OPTIONAL_INPUT_KEYS:
                if key in batch and key in input_names:
                    optional_inputs[key] = batch[key][i]

            make_contact_sheet(
                scene_dir / "contact_sheet.jpg",
                reference=reference,
                composite=composite,
                render=render,
                prediction=prediction,
                scene_id=scene_id,
                optional_inputs=optional_inputs,
            )

            if args.save_inputs:
                save_rgb_tensor(scene_dir / "reference.png", reference)
                save_rgb_tensor(scene_dir / "composite_in.png", composite)
                save_rgb_tensor(scene_dir / "render_in.png", render)
                for key, tensor in optional_inputs.items():
                    save_rgb_tensor(scene_dir / f"{key}_in.png", tensor)

            metrics = compute_metrics(prediction, reference, composite)
            row = {
                "scene_id": scene_id,
                "prediction": str(scene_dir / "prediction.png"),
                "contact_sheet": str(scene_dir / "contact_sheet.jpg"),
                "input_names": ",".join(input_names),
                **metrics,
            }
            rows_out.append(row)

            print(
                f"[DONE] {scene_id} "
                f"pred_l1={metrics['prediction_l1']:.6f} "
                f"comp_l1={metrics['composite_l1']:.6f} "
                f"gain={metrics['l1_improvement']:.6f}"
            )

    with (args.out_dir / "inference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=2, ensure_ascii=False)

    with (args.out_dir / "inference_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "scene_id",
            "input_names",
            "composite_l1",
            "prediction_l1",
            "l1_improvement",
            "composite_psnr",
            "prediction_psnr",
            "psnr_improvement",
            "prediction",
            "contact_sheet",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    if rows_out:
        mean_pred_l1 = sum(r["prediction_l1"] for r in rows_out) / len(rows_out)
        mean_comp_l1 = sum(r["composite_l1"] for r in rows_out) / len(rows_out)
        mean_gain = sum(r["l1_improvement"] for r in rows_out) / len(rows_out)
        print("[SUMMARY]")
        print("  scenes         :", len(rows_out))
        print("  mean comp L1   :", f"{mean_comp_l1:.6f}")
        print("  mean pred L1   :", f"{mean_pred_l1:.6f}")
        print("  mean L1 gain   :", f"{mean_gain:.6f}")

    print("[OUT]", args.out_dir)


if __name__ == "__main__":
    main()
