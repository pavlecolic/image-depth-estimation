from pathlib import Path

import matplotlib
import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=0.0, vmax=80.0, cmap="plasma"):
    value = value.detach().cpu().numpy()[0, :, :]
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmax - vmin > 1e-5:
        normalized = (value - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(value)

    colored = matplotlib.colormaps[cmap](normalized, bytes=True)
    return colored[:, :, :3].transpose((2, 0, 1))


def ensure_directories(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth.tar"):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filename,
    )


folder_structure = ensure_directories
