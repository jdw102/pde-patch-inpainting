from src.image_util import extract_rectangle
import cv2 as cv
import matlab.engine
import numpy as np


def pde_inpaint(image, mask, radius=3):
    return cv.inpaint(image, mask, inpaintRadius=radius, flags=cv.INPAINT_NS)


def texture_transfer(input_texture, target_image, w=10.0, al=0.43, i=1):
    eng = matlab.engine.start_matlab()
    path = "./transfer_script"
    eng.addpath(path, nargout=0)
    input_texture = np.ascontiguousarray(input_texture)
    target_image = np.ascontiguousarray(target_image)
    result = eng.transfer(input_texture.astype(float) / 255.0, target_image.astype(float) / 255.0, w, al, i, nargout=1)
    eng.quit()
    return np.array(result) * 255


def restore(damaged_image, mask, mask_rect, source_rect, patch_width=10.0, alpha=0.43, iterations=1):
    x, y, width, height = mask_rect
    restored_image = pde_inpaint(damaged_image, mask)
    initial_patch = extract_rectangle(restored_image, mask_rect)
    source_texture = extract_rectangle(restored_image, source_rect)
    texture_patch = texture_transfer(source_texture, initial_patch, patch_width, alpha, iterations)
    restored_image[y:y+height, x:x+width] = texture_patch
    return restored_image
