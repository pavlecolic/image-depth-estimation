import numpy as np
from PIL import Image


def is_pil_image(image):
    return isinstance(image, Image.Image)


def is_numpy_image(image):
    return isinstance(image, np.ndarray) and image.ndim in {2, 3}
