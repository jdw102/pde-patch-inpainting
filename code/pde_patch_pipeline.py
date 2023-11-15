from code.image_util import extract_rectangle
from code.texture_transfer import generate_texture_transfer
import cv2 as cv


def pde_inpaint(image, mask, radius=3):
    return cv.inpaint(image, mask, inpaintRadius=radius, flags=cv.INPAINT_NS)


def pde_patch(damaged_image, mask, mask_rect, source_rect):
    x, y, width, height = mask_rect
    restored_image = pde_inpaint(damaged_image, mask)
    initial_patch = extract_rectangle(restored_image, mask_rect)
    source_texture = extract_rectangle(restored_image, source_rect)
    textured_patch = generate_texture_transfer(source_texture, initial_patch)
    restored_image[y:y+height, x:x+width] = textured_patch
    return restored_image
