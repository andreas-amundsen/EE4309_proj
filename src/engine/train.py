# -*- coding: utf-8 -*-
import os, argparse, time
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm



from src.datasets.voc import VOCDataset, collate_fn
from src.utils.transforms import Compose, ToTensor, RandomHorizontalFlip
from src.models import build_model, AVAILABLE_MODELS
from src.utils.common import seed_everything, save_jsonl


def get_args():
    ap = argparse.ArgumentParser()

    # Default to using trainval for both; subset sizes control the split
    ap.add_argument("--train-set", type=str, default="trainval")
    ap.add_argument("--val-set", type=str, default="trainval")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default="vit")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--train-subset-size", type=int, default=None)
    ap.add_argument("--val-subset-size", type=int, default=None)

    return ap.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = Compose([RandomHorizontalFlip(0.5), ToTensor()])
    val_tf = Compose([ToTensor()])

    # If both use trainval dataset, split it into train and val
    if args.train_set == "trainval" and args.val_set == "trainval":
        # Load the complete trainval dataset
        full_dataset = VOCDataset(image_set="trainval", transforms=None)
        total_size = len(full_dataset)

        # Calculate split point (80% train, 20% val)
        if args.train_subset_size or args.val_subset_size:
            # Use specified subset sizes with bounds checking
            train_size = args.train_subset_size if args.train_subset_size else int(0.8 * total_size)
            val_size = args.val_subset_size if args.val_subset_size else int(0.2 * total_size)

            end_train = min(train_size, total_size)
            end_val = min(end_train + val_size, total_size)
            train_indices = range(0, end_train)
            val_indices = range(end_train, end_val)
        else:
            # Use 80/20 split of full dataset
            split_point = int(0.8 * total_size)
            train_indices = range(0, split_point)
            val_indices = range(split_point, total_size)

        # Create train and val subsets
        train_set = VOCDataset(image_set="trainval", transforms=train_tf)
        train_set = torch.utils.data.Subset(train_set, train_indices)
        val_set = VOCDataset(image_set="trainval", transforms=val_tf)
        val_set = torch.utils.data.Subset(val_set, val_indices)
    else:
        # Use original logic (train_set and val_set from different datasets)
        base_train = VOCDataset(image_set=args.train_set, transforms=train_tf)
        if args.train_subset_size:
            end_train = min(args.train_subset_size, len(base_train))
            train_set = torch.utils.data.Subset(base_train, range(end_train))
        else:
            train_set = base_train
        base_val = VOCDataset(image_set=args.val_set, transforms=val_tf)
        if args.val_subset_size:
            end_val = min(args.val_subset_size, len(base_val))
            val_set = torch.utils.data.Subset(base_val, range(end_val))
        else:
            val_set = base_val
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Resolve default output directory after parsing to depend on the model name.
    output_dir = args.output or f"runs/{args.model}_voc07"
    args.output = str(output_dir)

    num_classes = 21  # VOC 20 classes + background
    model = build_model(args.model, num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    sched = StepLR(optim, step_size=6, gamma=0.1)

    # Only enable AMP if CUDA is available and not explicitly disabled
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, ncols=100, desc=f"train[{epoch}/{args.epochs}]")
        loss_sum = 0.0

        for images, targets in pbar:
            # ===== STUDENT TODO: Implement training step =====
            # Hint: Complete the training loop:
            # 1. Use autocast context for mixed precision if enabled
            # 2. Forward pass: get loss_dict from model(images, targets)
            # 3. Sum all losses from the loss dictionary
            # 4. Backward pass: scale losses, compute gradients, step optimizer
            # 5. Update scaler for mixed precision training
            images = [img.to(device) for img in images]
            targets = [
                {k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()}
                for t in targets
            ]

            # print("Len of targets:", len(targets))
            # print("Targets object", targets)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss_dict = model(images, targets)          # dict of losses
                # print("\nLoss dict:", loss_dict)
                losses = sum(loss_dict.values())

           # ======================================================
            # 🔍 DEBUG: inspect anchor generation for one sample
            # ======================================================
            # if epoch == 1:
            #     model.eval()
            #     with torch.no_grad():
            #         # 1. Transform and get features exactly as RPN expects
            #         images_t = model.transform(images)
            #         features = model.backbone(images_t.tensors)
            #         if isinstance(features, dict):
            #             features = list(features.values())

            #         # 2. Generate anchors using both image and features
            #         anchors = model.rpn.anchor_generator(images_t, features)
            #         print(f"[Anchor Debug] Generated anchors for {len(anchors)} images.")
            #         print(f"[Anchor Debug] Image 0: {anchors[0].shape}")
            #         print(f"[Anchor Debug] Coord range: {anchors[0].min().item():.1f} → {anchors[0].max().item():.1f}")
            #     model.train()
            # ======================================================

            scaler.scale(losses).backward()
            scaler.step(optim)
            scaler.update()
            # raise NotImplementedError("Training step not implemented")
            # ==================================================

            loss_sum += losses.item()
            pbar.set_postfix(loss=f"{losses.item():.3f}")

        sched.step()
        avg_loss = loss_sum / len(train_loader)
        save_jsonl([{"epoch": epoch, "loss": avg_loss}], os.path.join(args.output, "logs.jsonl"))


        # ===== STUDENT TODO: Implement mAP evaluation =====
        # Hint: Implement validation loop to compute mAP@0.5:
        # 1. Import and initialize MeanAveragePrecision from torchmetrics
        # 2. Set model to eval mode and disable gradients
        # 3. Loop through validation data:
        #    - Move images to device
        #    - Get model predictions (no targets needed for inference)
        #    - Update metric with predictions and ground truth targets
        # 4. Compute final mAP and extract the "map" value
        # Handle exceptions gracefully and set map50 = -1.0 if evaluation fails
        try:
          from torchmetrics.detection.mean_ap import MeanAveragePrecision

          model.eval()
          metric = MeanAveragePrecision(iou_type="bbox")
          with torch.no_grad():
              for images, targets in val_loader:
                    images = [img.to(device) for img in images]

                    # Inference
                    outputs = model(images)

                    # torchmetrics expects CPU tensors
                    preds = []
                    for o in outputs:
                        preds.append({
                            "boxes": o["boxes"].detach().cpu(),
                            "scores": o["scores"].detach().cpu(),
                            "labels": o["labels"].detach().cpu(),
                        })

                    gts = []
                    for t in targets:
                        gts.append({
                            "boxes": t["boxes"].detach().cpu(),
                            "labels": t["labels"].detach().cpu(),
                        })

                    metric.update(preds, gts)

          metrics = metric.compute()
          map50 = float(metrics.get("map_50", torch.tensor(-1.0)).item())
        except Exception as e:
          print("Eval skipped due to:", e)
          map50 = -1.0

        # ====================================================

        is_best = map50 > best_map
        best_map = max(best_map, map50)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "best_map": best_map,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.output, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output, "best.pt"))
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}  mAP@0.5={map50:.4f}  best={best_map:.4f}")

    ### MAP for training data

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        with torch.no_grad():
            for images, targets in train_loader:
                images = [img.to(device) for img in images]

                # Inference
                outputs = model(images)

                # torchmetrics expects CPU tensors
                preds = []
                for o in outputs:
                    preds.append({
                        "boxes": o["boxes"].detach().cpu(),
                        "scores": o["scores"].detach().cpu(),
                        "labels": o["labels"].detach().cpu(),
                    })

                print("\n\nOutputs from training data", outputs)

                gts = []
                for t in targets:
                    gts.append({
                        "boxes": t["boxes"].detach().cpu(),
                        "labels": t["labels"].detach().cpu(),
                    })

                metric.update(preds, gts)

        metrics = metric.compute()
        map50 = float(metrics.get("map_50", torch.tensor(-1.0)).item())
        print("\n\nMap on training images:", map50)
    except Exception as e:
        print("Eval skipped due to:", e)
        map50 = -1.0

    ## Added by Andreas for debugging
    from torchvision.transforms.functional import to_pil_image, resize
    import random
    import torchvision

    # pick 5 random indices
    num_samples = 5
    indices = list(range(len(train_set) - num_samples, len(train_set)))

    vis_dir = Path(args.output) / "train_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        img, target = train_set[idx]
        img_tensor = img.to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)[0]

        img_cpu = img.detach().cpu()
        pred = {k: v.detach().cpu() for k, v in output.items()}
        gt = {k: v.detach().cpu() for k, v in target.items()}

        # resize and convert to uint8
        canvas = (img_cpu * 255).byte()

        # draw ground-truth boxes
        if gt["boxes"].numel() > 0:
            canvas = torchvision.utils.draw_bounding_boxes(
                canvas, gt["boxes"],
                labels=["GT"] * gt["boxes"].shape[0],
                colors="green", width=2
            )

        # draw predicted boxes
        if pred["boxes"].numel() > 0:
            boxes = pred["boxes"]
            scores = pred["scores"]
            canvas = torchvision.utils.draw_bounding_boxes(
                canvas, boxes,
                labels=[f"{s:.2f}" for s in scores],
                colors="red", width=2
            )
        
        # print("Checking test boxes for evaluation")

        # if len(pred["boxes"]):
        #     top_boxes = pred["boxes"][:3]  # print first 3 boxes for readability
        #     top_scores = pred["scores"][:3]
        #     print(f"[{i+1}] preds={len(pred['boxes'])}, "
        #         f"max_score={pred['scores'].max().item():.2f}, "
        #         f"sample_boxes={[b.tolist() for b in top_boxes]}, "
        #         f"sample_scores={[round(s.item(), 2) for s in top_scores]}")
        # else:
        #     print(f"[{i+1}] preds=0")


        out_path = vis_dir / f"train_sample_{i+1}_vis.jpg"
        to_pil_image(canvas).save(out_path)
        print(f"✅ Saved training visualization to {out_path}")

        # ===============================

if __name__ == "__main__":
    main()
