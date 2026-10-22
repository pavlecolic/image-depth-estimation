import argparse
import configparser
import datetime
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from depth_estimation.data.kitti import get_data, split_train_loader
from depth_estimation.evaluation.validation import validate
from depth_estimation.models import PTModel
from depth_estimation.training.losses import BerHuLoss, gradient_x, gradient_y, ssim
from depth_estimation.utils import AverageMeter, ensure_directories, save_checkpoint
from depth_estimation.visualization import log_progress, plot_training_curves


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train DenseNet-169 monocular depth estimation on KITTI.")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
    parser.add_argument("--lr", "--learning-rate", default=0.0001, type=float, help="Initial learning rate.")
    parser.add_argument("--bs", "--batch-size", default=4, type=int, help="Batch size.")
    parser.add_argument("--dataset-root", default=Path("../Datasets/kitti_kaggle"), type=Path, help="KITTI root with train and test folders.")
    parser.add_argument("--config", default=Path("configs/default.ini"), type=Path, help="Training config file.")
    parser.add_argument("--model-dir", default=Path("models_kitti_1"), type=Path, help="Checkpoint output directory.")
    parser.add_argument("--runs-dir", default=Path("runs"), type=Path, help="TensorBoard output directory.")
    parser.add_argument("--val-ratio", default=0.02, type=float, help="Validation split ratio from the training loader.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader worker count.")
    parser.add_argument("--launch-tensorboard", action="store_true", help="Start TensorBoard on port 6006 when it is not running.")
    return parser.parse_args()


def maybe_launch_tensorboard(logdir):
    if not is_port_in_use(6006):
        subprocess.Popen(["tensorboard", "--logdir", str(logdir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_directories(args.model_dir, args.runs_dir)

    model = PTModel()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    data_loader = get_data(batch_size=args.bs, mode="train", dataset_root=args.dataset_root, num_workers=args.num_workers)
    train_loader, validation_loader = split_train_loader(data_loader, val_ratio=args.val_ratio)

    l1_criterion = nn.L1Loss()
    berhu = BerHuLoss()
    ssim_val_range = config.getfloat("variables", "ssim_val_range", fallback=80.0)
    depth_weight = config.getfloat("variables", "l_depth_lambda", fallback=0.1)
    gradient_weight = config.getfloat("variables", "l_grad_lambda", fallback=1.0)
    tensorboard_dir = config.get("variables", "tensorboard_dir", fallback="tensorboard_logs_densenet_169")

    if args.launch_tensorboard:
        maybe_launch_tensorboard(args.runs_dir)

    prefix = f"densenet_{args.bs}"
    writer = SummaryWriter(
        log_dir=str(args.runs_dir / tensorboard_dir),
        comment=f"{prefix}-lr{args.lr}-e{args.epochs}-bs{args.bs}",
        flush_secs=30,
    )

    epoch_losses = []
    batch_times = []
    start = time.time()

    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        num_batches = len(train_loader)

        for index, sample in enumerate(train_loader):
            optimizer.zero_grad()
            image = sample["img"].to(device)
            depth = sample["depth_interp"].to(device, non_blocking=True)
            output = model(image)

            if output.shape[-2:] != depth.shape[-2:]:
                output = F.interpolate(output, size=depth.shape[-2:], mode="bilinear", align_corners=False)

            l_depth = berhu(output, depth)
            l_ssim = torch.clamp((1 - ssim(output, depth, val_range=ssim_val_range)) * 0.5, 0, 1)

            pred_dx = gradient_x(output)
            pred_dy = gradient_y(output)
            gt_dx = gradient_x(depth)
            gt_dy = gradient_y(depth)
            l_grad = l1_criterion(pred_dx, gt_dx) + l1_criterion(pred_dy, gt_dy)
            loss = l_ssim + depth_weight * l_depth + gradient_weight * l_grad

            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (num_batches - index))))
            step = epoch * num_batches + index

            if index % 5 == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]	Time {batch_time.val:.3f} ({batch_time.sum:.3f})	ETA {eta}	Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                        epoch, index, num_batches, batch_time=batch_time, loss=losses, eta=eta
                    )
                )
                writer.add_scalar("Train/Loss", losses.val, step)

            if index % 300 == 0:
                log_progress(model, writer, validation_loader, step, device=device)

        epoch_losses.append(losses.avg)
        batch_times.append(batch_time.avg)
        save_checkpoint(model, optimizer, epoch, losses.avg, filename=args.model_dir / f"checkpoint_epoch_{epoch}.pth.tar")

        validation_metrics = validate(model, validation_loader, device)
        print(f"Validation Metrics @ Epoch {epoch}:")
        for key, value in validation_metrics.items():
            print(f"{key}: {value:.4f}")
            writer.add_scalar(f"Val/{key}", value, epoch)

        log_progress(model, writer, validation_loader, epoch, device=device)
        writer.add_scalar("Train/Loss.avg", losses.avg, epoch)
        scheduler.step()
        writer.add_scalar("Train/Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

    elapsed_time = time.time() - start
    plot_training_curves(epoch_losses, batch_times)
    writer.close()
    print(f"Training complete. Total time: {elapsed_time:.2f} sec")


if __name__ == "__main__":
    main()
