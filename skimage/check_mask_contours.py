from pathlib import Path
import json

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
    if len(contours) != 1:
        raise ValueError(f'Contours length is not one line. Contours count={len(contours)}')
    contour = contours[0]
    point1, point2 = (contour[0], contour[-1])
    return img, point1, point2


# def draw_dot(ax, point):
#     y, x = tuple(point)
#     c = patches.Circle(xy=(x, y), radius=4, color='red')
#     ax.add_patch(c)
def to_num(name):
    num = name.split("_")[1]
    return int(num)

def main():
    def _iter_contour_checked_number():
        for i, (image_path, mask_path) in enumerate(iter_image_mask_paths()):
            try:
                img, point1, point2 = load_mask(mask_path)
                #print(points)
                n = to_num(image_path.stem)
                assert n ==  to_num(mask_path.stem)
                #print(n, image_path.stem, mask_path.stem)
                yield n
            except ValueError as e:
                #print(f"Error[{e}]: {image_path.name}")
                pass
    nums = list(_iter_contour_checked_number())
    j = json.dumps(nums, indent=4)
    Path("contour_checked_numbers.json").write_text(j)


if __name__ == "__main__":
    main()
