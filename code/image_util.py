import numpy as np
import cv2 as cv


def load_damaged_image(damage_rect, image_name):
    image = cv.imread(image_name)
    # Create a mask to represent the damaged area
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv.rectangle(mask, (damage_rect[0], damage_rect[1]),
                 (damage_rect[0] + damage_rect[2], damage_rect[1] + damage_rect[3]), 255, -1)
    # Damage the image by turning the patch black
    image[damage_rect[1]:damage_rect[1] + damage_rect[3], damage_rect[0]:damage_rect[0] + damage_rect[2]] = 0
    return image, mask


def load_image(image_name):
    return cv.imread(image_name)


def extract_rectangle(image, rect):
    x, y, height, width = rect
    print(x, y, height, width)
    return image[y:y + height, x:x + width]
