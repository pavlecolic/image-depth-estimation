from pathlib import Path

import numpy as np
from PIL import Image

from .fill_depth import fill_depth_colorization
from .interpolate import interpdepth


class KittiLoader:
    def __init__(self, kitti_dir, mode, cam=2):
        if cam not in {2, 3}:
            raise ValueError("Camera id must be 2 or 3")

        self.mode = mode
        self.cam = cam
        self.files = []
        self.shared_idx = []
        self.kitti_root = Path(kitti_dir)

        current_path = Path(__file__).resolve().parent
        filepath = current_path / "filenames" / f"eigen_{self.mode}_files.txt"
        shared_path = current_path / "filenames" / "eigen692_652_shared_index.txt"

        with filepath.open("r", encoding="utf-8") as file:
            for row in file.read().splitlines():
                if not row:
                    continue
                data_info = row.split(" ")
                data_idx_select = 0 if cam == 2 else 1
                self.files.append({"rgb": data_info[data_idx_select], "depth": data_info[data_idx_select + 1]})

        with shared_path.open("r", encoding="utf-8") as file:
            self.shared_idx = [int(item) for item in file.read().splitlines() if item]

    def shared_index(self):
        return self.shared_idx

    def data_length(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = self.kitti_root / filename.replace("\\", "/")
        if not file_path.exists():
            raise FileNotFoundError(f"{err_info}: {file_path}")
        return file_path

    @staticmethod
    def _read_depth(depth_path):
        depth_png = np.array(Image.open(depth_path), dtype=int)
        if np.max(depth_png) <= 255:
            raise ValueError(f"Expected a 16-bit KITTI depth map: {depth_path}")
        return depth_png.astype(np.float32) / 256.0

    def _read_data(self, item_files, interp_method):
        rgb_path = self._check_path(item_files["rgb"], "Cannot find RGB image")
        depth_path = self._check_path(item_files["depth"], "Cannot find depth file")

        rgb = Image.open(rgb_path).convert("RGB")
        depth = self._read_depth(depth_path)
        data = {"img": rgb, "depth": depth}

        if interp_method == "linear":
            data["depth_interp"] = interpdepth(depth)
        elif interp_method == "nyu":
            image_gray_arr = np.array(rgb.convert("L"))
            data["depth_interp"] = fill_depth_colorization(image_gray_arr, depth)
        elif interp_method != "nop":
            raise ValueError("Invalid interp_method parameter")

        return data

    def load_item(self, idx, interp_method="nop"):
        return self._read_data(self.files[idx], interp_method)


Kittiloader = KittiLoader
