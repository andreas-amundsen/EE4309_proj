# -*- coding: utf-8 -*-
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.voc import VOCDataset, collate_fn
from src.utils.transforms import Compose, ToTensor
from src.models import build_model, AVAILABLE_MODELS


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-set", type=str, default="test")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--subset-size", type=int, default=None)
    ap.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default=None)
    return ap.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VOCDataset(image_set=args.image_set, transforms=Compose([ToTensor()]))
    if args.subset_size:
        ds = torch.utils.data.Subset(ds, range(args.subset_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    num_classes = 21
    ckpt = torch.load(args.weights, map_location="cpu")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    saved_model = None
    if isinstance(saved_args, dict):
        saved_model = saved_args.get("model")
    model_name = (args.model or saved_model or "vit").lower()
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {AVAILABLE_MODELS}")
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)

    with torch.no_grad():
        for images, targets in tqdm(dl, ncols=100, desc="eval"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            # accumulate metrics for all items in the batch
            for out, tgt in zip(outputs, targets):
                metric.update(
                    [{k: v.cpu() for k, v in out.items()}],
                    [{k: v.cpu() for k, v in tgt.items()}],
                )
    res = metric.compute()
    print("mAP@0.5:", float(res["map"]))
    if "map_per_class" in res and res["map_per_class"] is not None:
        print("per-class AP:")
        for ap in res["map_per_class"].tolist():
            print(f"{ap:.3f}", end=" ")
        print()


if __name__ == "__main__":
    main()
