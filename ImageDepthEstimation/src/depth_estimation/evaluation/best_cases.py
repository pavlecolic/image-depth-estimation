import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from depth_estimation.data.kitti import get_testing_data
from depth_estimation.evaluation.metrics import evaluate_model, load_checkpoint
from depth_estimation.evaluation.validation import compute_depth_metrics


def save_best_cases(model, test_loader, device, save_dir, limit=10):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cases = []

    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            image = batch["img"].to(device)
            gt_depth = batch["depth"].to(device)
            pred_depth = model(image)
            if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
            metrics = compute_depth_metrics(gt_depth, pred_depth)
            cases.append((metrics["rel"], image.cpu(), gt_depth.cpu(), pred_depth.cpu(), index))

    for rank, (rel, image, gt_depth, pred_depth, index) in enumerate(sorted(cases, key=lambda item: item[0])[:limit]):
        TF.to_pil_image(image.squeeze(0)).save(save_dir / f"best_{rank:02d}_sample_{index}_rel_{rel:.3f}.png")
        np.save(save_dir / f"best_{rank:02d}_gt.npy", gt_depth.squeeze(0).numpy())
        np.save(save_dir / f"best_{rank:02d}_pred.npy", pred_depth.squeeze(0).numpy())


def parse_args():
    parser = argparse.ArgumentParser(description="Save the lowest-REL qualitative KITTI examples for a checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a .pth.tar checkpoint.")
    parser.add_argument("--dataset-root", default="../Datasets/kitti_kaggle", type=Path, help="KITTI root with train and test folders.")
    parser.add_argument("--save-dir", default=Path("best_cases"), type=Path, help="Output directory.")
    parser.add_argument("--limit", default=10, type=int, help="Number of cases to save.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    test_loader = get_testing_data(args.dataset_root)
    save_best_cases(model, test_loader, device, args.save_dir, args.limit)
    metrics = evaluate_model(model, test_loader, device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
