import numpy as np
import torch
import torch.nn.functional as F


def compute_depth_metrics(gt, pred):
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    valid_mask = (gt > 0) & (pred > 0)

    if not np.any(valid_mask):
        return {"rel": np.nan, "rmse": np.nan, "log10": np.nan, "delta_1": np.nan, "delta_2": np.nan, "delta_3": np.nan}

    gt = gt[valid_mask]
    pred = pred[valid_mask]
    ratio = np.maximum(gt / pred, pred / gt)

    return {
        "rel": np.mean(np.abs(gt - pred) / gt),
        "rmse": np.sqrt(np.mean((gt - pred) ** 2)),
        "log10": np.mean(np.abs(np.log10(gt) - np.log10(pred))),
        "delta_1": np.mean(ratio < 1.25),
        "delta_2": np.mean(ratio < 1.25 ** 2),
        "delta_3": np.mean(ratio < 1.25 ** 3),
    }


def validate(model, validation_loader, device):
    model.eval()
    metrics = []

    with torch.no_grad():
        for sample in validation_loader:
            image = sample["img"].to(device)
            depth = sample["depth_interp"].to(device)
            output = model(image)

            if output.shape[-2:] != depth.shape[-2:]:
                output = F.interpolate(output, size=depth.shape[-2:], mode="bilinear", align_corners=False)

            for pred, gt in zip(output, depth):
                metrics.append(compute_depth_metrics(gt, pred))

    return {key: np.nanmean([item[key] for item in metrics]) for key in metrics[0]}
