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
    restored_image = pde_inpaint(damaged_image, cv.cvtColor(mask, cv.COLOR_BGR2GRAY))
    cv.imshow("Initial Restore", restored_image)

    initial_patch = extract_rectangle(restored_image, mask_rect)
    source_texture = create_texture_patch(mask, restored_image)
    # source_texture = extract_rectangle(restored_image, source_rect)
    cv.imshow("Source Texture", source_texture)
    cv.waitKey(0)
    texture_patch = texture_transfer(source_texture, initial_patch, patch_width, alpha, iterations)
    restored_image[y:y+height, x:x+width] = texture_patch
    return restored_image


def square_border(mask, scale=1.5):
    nonzero_pixels = np.column_stack(np.where(mask > 0))
    min_coords = np.min(nonzero_pixels, axis=0)
    max_coords = np.max(nonzero_pixels, axis=0)
    min_x, min_y = min_coords[1], min_coords[0]
    max_x, max_y = max_coords[1], max_coords[0]
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    new_width = int((max_x - min_x) * scale)
    new_height = int((max_y - min_y) * scale)
    new_min_x = center_x - new_width // 2
    new_min_y = center_y - new_height // 2
    new_max_x = center_x + new_width // 2
    new_max_y = center_y + new_height // 2
    final = np.zeros_like(mask)
    cv.rectangle(final, (new_min_x, new_min_y), (new_max_x, new_max_y), (255, 255, 255), thickness=cv.FILLED)
    cv.rectangle(final, (min_x, min_y), (max_x, max_y), (0, 0, 0), thickness=cv.FILLED)
    return final, (new_min_x, new_min_y, new_max_x - new_min_x, new_max_y - new_min_y)


def create_texture_patch(mask, image, scale=2.0):
    new_mask, mask_rect = square_border(mask, scale)
    new_image = cv.bitwise_and(new_mask, image)
    return extract_rectangle(new_image, mask_rect)
