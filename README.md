# Image Depth Estimation

Monocular depth estimation project developed as a Bachelor thesis, focused on NYU Depth V2 and KITTI experiments. The cleaned project code lives in the ImageDepthEstimation/ directory.

The model uses a pretrained DenseNet-169 encoder with a lightweight decoder and skip connections to predict dense depth from a single RGB image.

## Highlights

- DenseNet-169 transfer-learning backbone for monocular depth prediction
- KITTI Eigen-style train and test split support
- NYU Depth V2 zip-based loader retained for earlier experiments
- Training loss combining BerHu depth loss, SSIM, and depth-gradient consistency
- Evaluation metrics: REL, RMSE, log10, and delta accuracy thresholds
- TensorBoard logging for losses, validation metrics, predictions, and error maps

## Repository Structure

~~~text
ImageDepthEstimation/
  configs/                       Training defaults
  scripts/                       Dataset utility scripts
  src/depth_estimation/
    data/                        NYU and KITTI data loading
    evaluation/                  Metrics and qualitative export scripts
    models/                      DenseNet-169 encoder-decoder model
    training/                    Training loop and loss functions
    visualization/               TensorBoard and curve plotting helpers
~~~

Exploratory notebooks, TensorBoard event files, caches, local datasets, checkpoints, and thesis document artifacts are intentionally excluded from this branch.

## Setup

Use Python 3.10 or newer. Install a PyTorch build that matches your CUDA setup, then install the project dependencies:

~~~bash
cd ImageDepthEstimation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
~~~

If you do not install the package in editable mode, run commands with PYTHONPATH=src from inside ImageDepthEstimation/.

## Data Layout

The KITTI loader expects this layout by default:

~~~text
../Datasets/kitti_kaggle/
  train/
  test/
~~~

NYU Depth V2 zip paths and normalization defaults are configured in ImageDepthEstimation/configs/default.ini.

## Training

~~~bash
cd ImageDepthEstimation
python -m depth_estimation.training.train --dataset-root ../Datasets/kitti_kaggle --epochs 15 --bs 4 --config configs/default.ini
~~~

## Evaluation

~~~bash
cd ImageDepthEstimation
python -m depth_estimation.evaluation.metrics --checkpoint models_kitti_1/checkpoint_epoch_14.pth.tar --dataset-root ../Datasets/kitti_kaggle
~~~

## Notes

Large data files, experiment notebooks, event logs, and model weights should stay outside git and be referenced from local paths or release artifacts.
