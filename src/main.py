import cv2 as cv
from src.image_util import load_damaged_image
from src.restore import restore
import numpy as np


def test(damage_rect, image_name, patch_width=10.0, alpha=0.43, iterations=1, texture_radius=40, inpainting_radius=10, algorithm=cv.INPAINT_NS):
    damaged_image, mask = load_damaged_image(damage_rect, image_name)
    restored_image = restore(damaged_image, mask, damage_rect, patch_width, alpha, iterations, texture_radius, inpainting_radius, algorithm)
    cv.imshow("Damaged Image", damaged_image)
    cv.imshow("Restored Image", restored_image)
    cv.waitKey(0)


if __name__ == '__main__':

    # test((300, 250, 100, 100), "../data/Granite-Rock.jpg")
    test((300, 250, 100, 100), "../data/Granite-Rock.jpg")
