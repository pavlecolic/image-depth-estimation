from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    values = [exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(values)
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    padd = 0
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    contrast_sensitivity = torch.mean(v1 / v2)
    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        score = ssim_map.mean()
    else:
        score = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return score, contrast_sensitivity
    return score


def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


class BerHuLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        c = torch.clamp(0.2 * torch.max(error), min=1e-6)
        return torch.where(error <= c, error, (error ** 2 + c ** 2) / (2 * c)).mean()
