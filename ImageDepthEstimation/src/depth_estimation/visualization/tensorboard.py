import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.utils as vutils

from depth_estimation.utils import colorize


def log_progress(model, writer, val_loader, step, device):
    model.eval()
    with torch.no_grad():
        sample = next(iter(val_loader))
        image = sample["img"].to(device)
        depth_key = "depth_interp" if "depth_interp" in sample else "depth"
        depth = sample[depth_key].to(device, non_blocking=True)

        if step == 0:
            writer.add_image("Train.1.Image", vutils.make_grid(image.data, nrow=6, normalize=True), step)
            writer.add_image("Train.2.Depth", colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), step)

        output = model(image)
        writer.add_image("Train.3.Ours", colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), step)
        writer.add_image("Train.4.Diff", colorize(vutils.make_grid(torch.abs(output - depth).data, nrow=6, normalize=False)), step)


def plot_training_curves(losses, batch_times, output_path="training_curves.png"):
    epochs = np.arange(len(losses))
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs, y=losses, marker="o", color="red")
    plt.title("Epoch-wise Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs, y=batch_times, marker="o", color="blue")
    plt.title("Batch Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


LogProgress = log_progress
