import argparse
from pathlib import Path


def build_split_file(dataset_root, output_txt, image_subpath="image_02/data", depth_subpath="proj_depth/groundtruth/image_02"):
    dataset_root = Path(dataset_root)
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with output_txt.open("w", encoding="utf-8") as file:
        for date_folder in sorted(dataset_root.iterdir()):
            if not date_folder.is_dir():
                continue

            for drive_folder in sorted(date_folder.iterdir()):
                if not drive_folder.is_dir():
                    continue

                image_dir = drive_folder / image_subpath
                depth_dir = drive_folder / depth_subpath
                if not image_dir.is_dir() or not depth_dir.is_dir():
                    continue

                for image_file in sorted(image_dir.glob("*.png")):
                    depth_file = depth_dir / image_file.name
                    if depth_file.exists():
                        image_rel = image_file.relative_to(dataset_root)
                        depth_rel = depth_file.relative_to(dataset_root)
                        file.write(f"{image_rel.as_posix()} {depth_rel.as_posix()}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a KITTI Eigen-style split file.")
    parser.add_argument("dataset_root", type=Path, help="KITTI split root to scan.")
    parser.add_argument("output_txt", type=Path, help="Output text file.")
    parser.add_argument("--image-subpath", default="image_02/data", help="Image folder inside each drive.")
    parser.add_argument("--depth-subpath", default="proj_depth/groundtruth/image_02", help="Depth folder inside each drive.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_split_file(args.dataset_root, args.output_txt, args.image_subpath, args.depth_subpath)


if __name__ == "__main__":
    main()
