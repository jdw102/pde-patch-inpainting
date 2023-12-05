from src.image_util import extract_rectangle
import cv2 as cv
import matlab.engine
import numpy as np


def transfer_restore(damaged_image, mask, mask_rect, name="", save_initial=False, patch_width=8.0, alpha=0.43, iterations=1.0, texture_radius=20, inpaint_radius=3, inpaint_algorithm=cv.INPAINT_NS):
    x, y, width, height = mask_rect
    restored_image = pde_inpaint(damaged_image, cv.cvtColor(mask, cv.COLOR_BGR2GRAY), inpaint_radius, inpaint_algorithm)
    if save_initial:
        cv.imwrite(f"../data/results/{name}-pde-tr{texture_radius}-alg{inpaint_algorithm == cv.INPAINT_NS}.jpg", restored_image)
    initial_patch = create_input_patch(mask, restored_image, int(patch_width))
    source_texture = create_texture_patch(mask, restored_image, texture_radius)
    texture_patch = texture_transfer(source_texture, initial_patch, patch_width, alpha, iterations)
    extract_middle(texture_patch, restored_image[y:y+height, x:x+width])
    return restored_image


def synthesis_restore(damaged_image, mask_rect, texture_rect, patch_width=8.0):
    x, y, width, height = mask_rect
    source_texture = extract_rectangle(damaged_image, texture_rect)
    texture_patch = texture_synthesis(source_texture, patch_width, (width, height))
    damaged_image[y:y+height, x:x+width] = texture_patch
    return damaged_image

def pde_inpaint(image, mask, radius=3, algorithm=cv.INPAINT_NS):
    return cv.inpaint(image, mask, inpaintRadius=radius, flags=algorithm)


def texture_transfer(input_texture, target_image, width, alpha, iterations):
    eng = matlab.engine.start_matlab()
    path = "./transfer_script"
    eng.addpath(path, nargout=0)
    input_texture = np.ascontiguousarray(input_texture)
    target_image = np.ascontiguousarray(target_image)
    result = eng.transfer(input_texture.astype(float) / 255.0, target_image.astype(float) / 255.0, width, alpha, iterations, nargout=1)
    eng.quit()
    return np.array(result) * 255


def texture_synthesis(input_texture, width, size):
    eng = matlab.engine.start_matlab()
    path = "./transfer_script"
    eng.addpath(path, nargout=0)
    input_texture = np.ascontiguousarray(input_texture)
    result = eng.synthesis(input_texture.astype(float) / 255.0, width, size[0], size[1])
    eng.quit()
    return np.array(result) * 255

def create_texture_patch(mask, image, radius):
    new_mask, mask_rect = square_border(mask, radius)
    new_image = cv.bitwise_and(new_mask, image)
    return extract_rectangle(new_image, mask_rect)


def create_input_patch(mask, image, radius):
    new_mask, mask_rect = square_border(mask, radius)
    return extract_rectangle(image, mask_rect)


def square_border(mask, radius):
    nonzero_pixels = np.column_stack(np.where(mask > 0))
    min_coords = np.min(nonzero_pixels, axis=0)
    max_coords = np.max(nonzero_pixels, axis=0)
    min_x, min_y = min_coords[1], min_coords[0]
    max_x, max_y = max_coords[1], max_coords[0]
    new_min_x = max(0, min_x - radius)
    new_min_y = max(0, min_y - radius)
    new_max_x = min(mask.shape[1] - 1, max_x + radius)
    new_max_y = min(mask.shape[0] - 1, max_y + radius)
    final = np.zeros_like(mask)
    cv.rectangle(final, (new_min_x, new_min_y), (new_max_x, new_max_y), (255, 255, 255), thickness=cv.FILLED)
    cv.rectangle(final, (min_x, min_y), (max_x, max_y), (0, 0, 0), thickness=cv.FILLED)
    return final, (new_min_x, new_min_y, new_max_x - new_min_x, new_max_y - new_min_y)


def extract_middle(larger_matrix, smaller_matrix):
    larger_shape = np.array(larger_matrix.shape)
    smaller_shape = np.array(smaller_matrix.shape)
    start_idx = (larger_shape - smaller_shape) // 2
    end_idx = start_idx + smaller_shape
    middle_region = larger_matrix[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]
    smaller_matrix[:, :] = middle_region
