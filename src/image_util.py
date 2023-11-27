import numpy as np
import cv2 as cv
import os


def load_damaged_image(damage_rect, image_name):
    image = cv.imread(image_name)
    mask = np.zeros_like(image)
    cv.rectangle(mask, (damage_rect[0], damage_rect[1]),
                 (damage_rect[0] + damage_rect[2], damage_rect[1] + damage_rect[3]), (255, 255, 255), -1)
    image[damage_rect[1]:damage_rect[1] + damage_rect[3], damage_rect[0]:damage_rect[0] + damage_rect[2]] = 0
    return image, mask


def load_image(image_name):
    return cv.imread(image_name)


def save_image(image, relative_path):
    absolute_path = os.path.abspath(relative_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    cv.imwrite(absolute_path, image)


def extract_rectangle(image, rect):
    x, y, height, width = rect
    return image[y:y + height, x:x + width]
