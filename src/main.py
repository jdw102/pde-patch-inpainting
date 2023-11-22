import cv2 as cv
from src.image_util import load_damaged_image
from src.restore import restore


def test(damage_rect, texture_rectangle, image_name, patch_width=10.0, alpha=0.43, iterations=1):
    damaged_image, mask = load_damaged_image(damage_rect, image_name)
    restored_image = restore(damaged_image, mask, damage_rect, texture_rectangle, patch_width, alpha, iterations)
    cv.imshow("Damaged Image", damaged_image)
    cv.imshow("Restored Image", restored_image)
    cv.waitKey(0)


if __name__ == '__main__':
    test((290, 250, 100, 100), (100, 400, 100, 200), "../data/Granite-Rock.jpg", patch_width=8.0)
