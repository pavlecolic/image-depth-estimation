import torchvision.transforms as transforms

from . import methods as augmethods
from .pipeline import BaseTransformer


class CustomTransformer(BaseTransformer):
    def __init__(self, phase):
        super().__init__(phase)
        if self.phase not in {"train", "test", "val"}:
            raise ValueError("Invalid phase parameter")

    def get_joint_transform(self):
        if self.phase == "train":
            return transforms.Compose([
                augmethods.TransToPIL(),
                augmethods.Scale("pair", [384, 1280]),
                augmethods.RandomHorizontalFlip(),
            ])
        return transforms.Compose([augmethods.TransToPIL()])

    def get_img_transform(self):
        if self.phase == "train":
            return transforms.Compose([
                augmethods.ImgAug(),
                augmethods.ToTensor("Img"),
            ])
        return transforms.Compose([
            augmethods.Scale("Img", [384, 1280]),
            augmethods.ToTensor("Img"),
        ])

    def get_depth_transform(self):
        return transforms.Compose([augmethods.ToTensor("depth")])


Transformer = CustomTransformer
CustTransformer = CustomTransformer
