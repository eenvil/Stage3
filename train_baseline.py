from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from stage3_dataset import Stage3RelightDataset
from stage3_unet import ResidualUNet, count_parameters


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1).clamp(0, 1)


def save_visual_grid(batch: Dict, pred: torch.Tensor, out_path: Path, max_items: int = 4) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref = batch["reference"][:max_items]
    comp = batch["composite"][:max_items]
    render = batch["render"][:max_items]
    pred = pred[:max_items]

    before_err = torch.stack([abs_error(comp[i], ref[i]) for i in range(len(ref))])
    after_err = torch.stack([abs_error(pred[i], ref[i]) for i in range(len(ref))])

    cols = [
        ("reference", ref),
        ("composite_in", comp),
        ("render_in", render),
        ("prediction", pred),
        ("err_before", before_err),
        ("err_after", after_err),
    ]

    imgs = [[tensor_to_pil(tensor[i]) for _, tensor in cols] for i in range(len(ref))]

    w, h = imgs[0][0].size
    label_h = 24
    gap = 6
    canvas = Image.new("RGB", (len(cols) * w + (len(cols) + 1) * gap, len(imgs) * (h + label_h) + (len(imgs) + 1) * gap), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    y = gap
    for row_imgs in imgs:
        x = gap
        for (label, _), im in zip(cols, row_imgs):
            draw.text((x, y), label, fill=(0, 0, 0))
            canvas.paste(im, (x, y + label_h))
            x += w + gap
        y += h + label_h + gap

    canvas.save(out_path, quality=92)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl: DataLoader, device: torch.device, max_batches: int = 50) -> Dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_n = 0

    for i, batch in enumerate(dl):
        if i >= max_batches:
            break

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        comp = batch["composite"].to(device, non_blocking=True)

        pred = model(x, comp)
        l1 = F.l1_loss(pred, y, reduction="sum")
        mse = F.mse_loss(pred, y, reduction="sum")

        total_l1 += float(l1.item())
        total_mse += float(mse.item())
        total_n += y.numel()

    mean_l1 = total_l1 / max(1, total_n)
    mean_mse = total_mse / max(1, total_n)
    psnr = -10.0 * math.log10(max(mean_mse, 1e-12))

    return {
        "val_l1": mean_l1,
        "val_mse": mean_mse,
        "val_psnr": psnr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True, type=Path)
    ap.add_argument("--val-jsonl", required=True, type=Path)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--train-limit", type=int, default=-1)
    ap.add_argument("--val-limit", type=int, default=512)
    ap.add_argument("--no-albedo", action="store_true")
    ap.add_argument("--grad-loss-weight", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.out_dir / "config.json", vars(args) | {
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "out_dir": str(args.out_dir),
    })

    inputs = ["composite", "render"]
    if not args.no_albedo:
        inputs.append("albedo")

    train_ds = Stage3RelightDataset(
        jsonl_path=args.train_jsonl,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        input_names=inputs,
        target_name="reference",
        missing_optional="zeros",
        require_all_inputs=False,
        limit=args.train_limit,
        shuffle=True,
        seed=123,
    )
    val_ds = Stage3RelightDataset(
        jsonl_path=args.val_jsonl,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        input_names=inputs,
        target_name="reference",
        missing_optional="zeros",
        require_all_inputs=False,
        limit=args.val_limit,
        shuffle=False,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet(input_channels=train_ds.channels, base_channels=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    print("[INFO] device       :", device)
    print("[INFO] inputs       :", inputs)
    print("[INFO] channels     :", train_ds.channels)
    print("[INFO] train rows   :", len(train_ds), "skipped", train_ds.skipped)
    print("[INFO] val rows     :", len(val_ds), "skipped", val_ds.skipped)
    print("[INFO] params       :", count_parameters(model))

    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        running_l1 = 0.0
        running_grad = 0.0
        n_steps = 0

        for batch in train_dl:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            comp = batch["composite"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                pred = model(x, comp)
                l1 = F.l1_loss(pred, y)
                gl = gradient_loss(pred, y)
                loss = l1 + args.grad_loss_weight * gl

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            running_l1 += float(l1.item())
            running_grad += float(gl.item())
            n_steps += 1
            global_step += 1

        metrics = evaluate(model, val_dl, device)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": running / max(1, n_steps),
            "train_l1": running_l1 / max(1, n_steps),
            "train_grad": running_grad / max(1, n_steps),
            **metrics,
            "seconds": elapsed,
        }
        history.append(record)
        save_json(args.out_dir / "history.json", history)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train={record['train_loss']:.6f} "
            f"l1={record['train_l1']:.6f} "
            f"val_l1={record['val_l1']:.6f} "
            f"psnr={record['val_psnr']:.2f} "
            f"time={elapsed:.1f}s"
        )

        # Save visuals from first validation batch.
        batch = next(iter(val_dl))
        x = batch["x"].to(device)
        comp = batch["composite"].to(device)
        with torch.no_grad():
            pred = model(x, comp).cpu()
        save_visual_grid(batch, pred, args.out_dir / "visuals" / f"epoch_{epoch:03d}.jpg")

        if epoch % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "inputs": inputs,
                "channels": train_ds.channels,
                "base_channels": args.base_channels,
                "history": history,
            }
            torch.save(ckpt, args.out_dir / "checkpoint_last.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "inputs": inputs,
            "channels": train_ds.channels,
            "base_channels": args.base_channels,
            "history": history,
        },
        args.out_dir / "checkpoint_final.pt",
    )
    print("[DONE] Training complete:", args.out_dir)


if __name__ == "__main__":
    main()
