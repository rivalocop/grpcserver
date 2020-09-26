import io

import cv2
import numpy as np
from PIL import Image


def preprocess_raw_image(image):
    img = np.frombuffer(image, np.uint8).reshape(352, 288, 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (300, 300))
    # Image.fromarray(img).show()
    # img = cv2.flip(img, 1)
    return img


def preprocess_png_image(image):
    with io.BytesIO(image) as f:
        # Load it as numpy array
        image = np.array(Image.open(f))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (300, 300))
        return image
