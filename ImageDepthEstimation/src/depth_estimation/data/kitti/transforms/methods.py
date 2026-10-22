import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

from depth_estimation.data.common import is_numpy_image, is_pil_image

from .base import BaseMethod


class TransToPIL(BaseMethod):
    def __init__(self):
        super().__init__()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, data_item):
        self.set_data(data_item)
        if not self._is_pil_image(self.img):
            data_item["img"] = self.to_pil(self.img)
        return data_item


class Scale(BaseMethod):
    def __init__(self, mode, size):
        super().__init__(mode)
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            self.size = (int(size[0]), int(size[1]))
        else:
            raise ValueError("size must be an int or a pair of ints")

        self.resize_img = transforms.Resize(self.size, interpolation=Image.BILINEAR)

    @staticmethod
    def _resize_depth(depth_np, target_size):
        if depth_np.dtype != np.float32:
            depth_np = depth_np.astype(np.float32)
        return cv2.resize(depth_np, (int(target_size[1]), int(target_size[0])), interpolation=cv2.INTER_NEAREST)

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode in {"pair", "Img"}:
            if not isinstance(self.img, Image.Image):
                raise TypeError(f"Expected PIL Image for img, got {type(self.img)}")
            data_item["img"] = self.resize_img(self.img)

        if self.mode in {"pair", "depth"}:
            depth_size = (self.size[0] // 2, self.size[1] // 2) if self.mode == "pair" else self.size
            if not isinstance(self.depth, np.ndarray):
                raise TypeError(f"Expected numpy array for depth, got {type(self.depth)}")
            data_item["depth"] = self._resize_depth(self.depth, depth_size)

            if "depth_interp" in data_item:
                if not isinstance(self.depth_interp, np.ndarray):
                    raise TypeError(f"Expected numpy array for depth_interp, got {type(self.depth_interp)}")
                data_item["depth_interp"] = self._resize_depth(self.depth_interp, depth_size)

        return data_item


class RandomHorizontalFlip(BaseMethod):
    def __init__(self):
        super().__init__()

    def __call__(self, data_item):
        self.set_data(data_item)
        if random.random() < 0.5:
            data_item["img"] = self.img.transpose(Image.FLIP_LEFT_RIGHT)
            data_item["depth"] = np.fliplr(data_item["depth"]).copy()
            if "depth_interp" in data_item:
                data_item["depth_interp"] = np.fliplr(data_item["depth_interp"]).copy()
        return data_item


class ColorChannelPermutation(BaseMethod):
    def __init__(self, probability):
        from itertools import permutations

        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image = sample["img"]
        if not is_pil_image(image):
            raise TypeError(f"img should be PIL Image. Got {type(image)}")
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
            sample["img"] = image
        return sample


class ImgAug(BaseMethod):
    def __init__(self):
        super().__init__()

    @staticmethod
    def adjust_pil(pil):
        brightness = random.uniform(0.8, 1.0)
        contrast = random.uniform(0.8, 1.0)
        saturation = random.uniform(0.8, 1.0)
        pil = F.adjust_brightness(pil, brightness)
        pil = F.adjust_contrast(pil, contrast)
        return F.adjust_saturation(pil, saturation)

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item["img"] = self.adjust_pil(self.img)
        return data_item


class ToTensor(BaseMethod):
    def __init__(self, mode):
        super().__init__(mode=mode)
        self.to_image_tensor = transforms.ToTensor()

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode == "Img":
            data_item["img"] = self.to_image_tensor(self.img)
        elif self.mode == "depth":
            data_item["depth"] = self.to_tensor(self.depth)
            if "depth_interp" in data_item:
                data_item["depth_interp"] = self.to_tensor(self.depth_interp)
        elif self.mode == "both":
            data_item["img"] = self.to_image_tensor(self.img)
            data_item["depth"] = self.to_tensor(self.depth)
            if "depth_interp" in data_item:
                data_item["depth_interp"] = self.to_tensor(self.depth_interp)
        else:
            raise ValueError(f"Unknown tensor conversion mode: {self.mode}")

        return data_item

    @staticmethod
    def to_tensor(pic):
        if not (is_pil_image(pic) or is_numpy_image(pic)):
            raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(np.ascontiguousarray(pic))
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 3:
                img = img.permute(2, 0, 1)
            return img.float()

        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float()


class ImgNormalize(BaseMethod):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item["img"] = self.normalize(self.img)
        return data_item
