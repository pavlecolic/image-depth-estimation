import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from depth_estimation.data.kitti import get_testing_data
from depth_estimation.evaluation.validation import compute_depth_metrics
from depth_estimation.models import PTModel


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model = PTModel()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch["img"].to(device)
            gt_depth = batch["depth"].to(device)
            pred_depth = model(image)

            if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)

            for gt, pred in zip(gt_depth, pred_depth):
                metrics_list.append(compute_depth_metrics(gt, pred))

    return {key: np.nanmean([metrics[key] for metrics in metrics_list]) for key in metrics_list[0]}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained depth-estimation checkpoint on KITTI.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a .pth.tar checkpoint.")
    parser.add_argument("--dataset-root", default="../Datasets/kitti_kaggle", type=Path, help="KITTI root with train and test folders.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    test_loader = get_testing_data(args.dataset_root)
    metrics = evaluate_model(model, test_loader, device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
