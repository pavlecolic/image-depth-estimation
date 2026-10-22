from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from depth_estimation.data.kitti.dataloader import KittiLoader
from depth_estimation.data.kitti.dataset import DataGenerator
from depth_estimation.data.kitti.transforms import methods as augmethods


DEFAULT_KITTI_ROOT = Path("../Datasets/kitti_kaggle")


def get_train_test_by_index_augmented(index, dataset_root=DEFAULT_KITTI_ROOT):
    dataset_root = Path(dataset_root)
    train_loader = KittiLoader(dataset_root / "train", "train")
    test_loader = KittiLoader(dataset_root / "test", "test")

    train_item = train_loader.load_item(index, interp_method="linear")
    test_item = test_loader.load_item(index)

    transformer = transforms.Compose([
        augmethods.Scale("pair", [192, 640]),
        augmethods.RandomHorizontalFlip(),
        augmethods.ToTensor("both"),
    ])

    return transformer(train_item), transformer(test_item)


def get_train_test_by_index(index, dataset_root=DEFAULT_KITTI_ROOT):
    dataset_root = Path(dataset_root)
    train_loader = KittiLoader(dataset_root / "train", "train")
    test_loader = KittiLoader(dataset_root / "test", "test")
    train_item = train_loader.load_item(index, interp_method="linear")
    test_item = test_loader.load_item(index)
    return train_item, test_item


def split_train_loader(full_loader, val_ratio=0.02):
    dataset = full_loader.dataset
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=full_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=full_loader.batch_size, shuffle=False)
    return train_loader, val_loader


def get_data(batch_size, mode="train", dataset_root=DEFAULT_KITTI_ROOT, num_workers=0):
    datagen = DataGenerator(Path(dataset_root) / mode, mode)
    return datagen.create_data(batch_size=batch_size, nthreads=num_workers)


def get_testing_data(dataset_root=DEFAULT_KITTI_ROOT):
    datagen = DataGenerator(Path(dataset_root) / "test", "test")
    return datagen.create_data(batch_size=1)


getTrainTestByIndexAugmentedData = get_train_test_by_index_augmented
getTrainTestByIndex = get_train_test_by_index
split_train_loader_old = split_train_loader
getData = get_data
getTestingData = get_testing_data
