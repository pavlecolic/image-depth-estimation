import configparser
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from depth_estimation.data.common import is_numpy_image, is_pil_image


def load_zip_to_memory(zip_file):
    with ZipFile(zip_file) as input_zip:
        data = {name: input_zip.read(name) for name in input_zip.namelist()}

    nyu2_train = _read_split(data, "test_nyu_dataset/nyu_data/data/nyu2_train.csv")
    nyu2_test = _read_split(data, "test_nyu_dataset/nyu_data/data/nyu2_test.csv")
    return data, shuffle(nyu2_train, random_state=0), nyu2_test


def _read_split(data, csv_path):
    rows = data[csv_path].decode("utf-8").splitlines()
    parsed = [[item.strip() for item in row.split(",")] for row in rows if row]
    return [["test_nyu_dataset/nyu_data/" + item for item in row] for row in parsed]


class RandomHorizontalFlip:
    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]
        if not is_pil_image(image):
            raise TypeError(f"image should be PIL Image. Got {type(image)}")
        if not is_pil_image(depth):
            raise TypeError(f"depth should be PIL Image. Got {type(depth)}")
        if np.random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        return {"image": image, "depth": depth}


class RandomChannelPermutation:
    def __init__(self, probability):
        from itertools import permutations

        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]
        if not is_pil_image(image):
            raise TypeError(f"image should be PIL Image. Got {type(image)}")
        if np.random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[np.random.randint(0, len(self.indices))])])
        return {"image": image, "depth": depth}


class DepthDatasetMemory(Dataset):
    def __init__(self, data, nyu_dataset, transform=None):
        self.data = data
        self.nyu_dataset = nyu_dataset
        self.transform = transform

    def __getitem__(self, idx):
        image_path, depth_path = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[image_path]))
        depth = Image.open(BytesIO(self.data[depth_path]))
        sample = {"image": image, "depth": depth}
        if self.transform:
            return self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


class ToTensor:
    def __init__(self, is_test=False, normalization="meters"):
        self.is_test = is_test
        self.normalization = normalization

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]
        image = self.to_tensor(image)
        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        elif self.normalization == "meters":
            depth = self.to_tensor(depth).float()
            depth = depth * (10.0 - 0.4) + 0.4
        else:
            depth = self.to_tensor(depth).float()

        return {"image": image, "depth": depth}

    @staticmethod
    def to_tensor(pic):
        if not (is_pil_image(pic) or is_numpy_image(pic)):
            raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

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
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        return img


def get_no_transform(is_test=True, normalization="meters"):
    return transforms.Compose([ToTensor(is_test=is_test, normalization=normalization)])


def get_default_train_transform(normalization="meters"):
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelPermutation(0.5),
        ToTensor(normalization=normalization),
    ])


def split_train_loader(train_loader, val_percentage=0.1):
    dataset = train_loader.dataset
    val_size = int(len(dataset) * val_percentage)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader_new = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=True)
    return train_loader_new, val_loader


def get_training_testing_data(batch_size, config_path="configs/default.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)
    zip_path = Path(config.get("dataset", "test_nyu_dataset_dir"))
    normalization = config.get("dataset", "normalization", fallback="meters")

    data, nyu2_train, nyu2_test = load_zip_to_memory(zip_path)
    train_dataset = DepthDatasetMemory(data, nyu2_train, transform=get_default_train_transform(normalization))
    test_dataset = DepthDatasetMemory(data, nyu2_test, transform=get_no_transform(normalization=normalization))
    return DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(test_dataset, batch_size, shuffle=False)


loadZipToMem = load_zip_to_memory
depthDatasetMemory = DepthDatasetMemory
getNoTransform = get_no_transform
getDefaultTrainTransform = get_default_train_transform
getTrainingTestingData = get_training_testing_data
