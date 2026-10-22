from torch.utils.data import DataLoader, Dataset

from .dataloader import KittiLoader
from .transforms import Transformer


class KittiDataset(Dataset):
    def __init__(self, kitti_dir, mode, transform=None):
        self.mode = mode
        self.kitti_root = kitti_dir
        self.transform = transform
        self.kitti_loader = KittiLoader(kitti_dir, mode, cam=2)

    def __getitem__(self, idx):
        interp_method = "linear" if self.mode == "train" else "nop"
        data_item = self.kitti_loader.load_item(idx, interp_method=interp_method)
        if self.transform is not None:
            return self.transform(data_item)
        return data_item

    def __len__(self):
        return self.kitti_loader.data_length()


class DataGenerator:
    def __init__(self, kitti_dir, phase, high_gpu=True):
        if phase not in {"train", "test", "val"}:
            raise ValueError("Invalid phase parameter")

        self.phase = phase
        self.high_gpu = high_gpu
        transformer = Transformer(self.phase)
        self.dataset = KittiDataset(kitti_dir, phase, transformer.get_transform())

    def create_data(self, batch_size, nthreads=0):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.phase == "train",
            num_workers=nthreads,
            pin_memory=self.high_gpu,
        )
