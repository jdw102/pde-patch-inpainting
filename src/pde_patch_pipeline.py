from src.image_util import extract_rectangle
from src.texture_transfer import generate_texture_transfer
import cv2 as cv


def pde_inpaint(image, mask, radius=3):
    return cv.inpaint(image, mask, inpaintRadius=radius, flags=cv.INPAINT_NS)


def pde_patch(damaged_image, mask, mask_rect, source_rect):
    x, y, width, height = mask_rect
    restored_image = pde_inpaint(damaged_image, mask)
    initial_patch = extract_rectangle(restored_image, mask_rect)
    source_texture = extract_rectangle(restored_image, source_rect)
    cv.imshow("Original Image", damaged_image)
    cv.imshow("Initial Patch", initial_patch)
    cv.imshow("Source Texture", source_texture)
    cv.waitKey(0)
    generate_texture_transfer(source_texture, initial_patch)
    texture_patch = cv.imread("./data/texture_patch1.jpg")
    restored_image[y:y+height, x:x+width] = texture_patch
    cv.imshow("Restored Image", restored_image)
    cv.waitKey(0)
    return restored_image
