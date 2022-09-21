from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import measure


def iter_image_mask_paths():
    base = Path("../blender-for-finger-segmentation/")
    mask_path = base / "data2/training/masks"
    mask_path_iter = mask_path.glob("*.png")
    image_path = base / "data2/training/images"
    image_path_iter = image_path.glob("*.jpg")
    images = sorted(image_path_iter, key=lambda p: p.name)
    masks = sorted(mask_path_iter, key=lambda p: p.name)
    return zip(images, masks)


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_mask(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    contours = measure.find_contours(img, 1.5, mask=(img > 0))
    assert len(contours) == 1
    contour = contours[0]
    point1, point2 = (contour[0], contour[-1])
    return img, point1, point2


def draw_dot(ax, point):
    y, x = tuple(point)
    c = patches.Circle(xy=(x, y), radius=4, color='red')
    ax.add_patch(c)


def main():
    for i, (image_path, mask_path) in enumerate(iter_image_mask_paths()):
        print(image_path, mask_path)
        img, point1, point2 = load_mask(mask_path)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(load_image(image_path))
        ax[1].imshow(img)
        draw_dot(ax[1], point1)
        draw_dot(ax[1], point2)
        plt.show()
        if i == 3:
            break


if __name__ == "__main__":
    main()
