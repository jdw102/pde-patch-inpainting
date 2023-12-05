import cv2 as cv
from matplotlib import pyplot as plt

from src.image_util import load_damaged_image, extract_rectangle
from src.restore import transfer_restore, synthesis_restore
import numpy as np


def test(damage_rect, image_name, name,
         patch_width=8.0,
         alpha=0.43,
         iterations=1.0,
         texture_radius=20,
         inpainting_radius=3,
         algorithm=cv.INPAINT_NS,
         synth_rect=None):
    damaged_image, mask, original = load_damaged_image(damage_rect, image_name)
    cv.imwrite(f"../data/results/{name}-damaged.jpg", damaged_image)
    restored_image = transfer_restore(damaged_image, mask, damage_rect, name, True, patch_width, alpha, iterations,
                                      texture_radius, inpainting_radius, algorithm)
    if synth_rect is not None:
        synth_image = synthesis_restore(damaged_image, damage_rect, synth_rect, patch_width)
        cv.imwrite(f"../data/results/{name}-synthesis.jpg", synth_image)
    path = f"../data/results/{name}-w{patch_width}-al{alpha}-iter{iterations}-tr{texture_radius}-ir{inpainting_radius}-alg{algorithm == cv.INPAINT_NS}.jpg"
    cv.imwrite(path, restored_image)
    return calculate_error(extract_rectangle(cv.imread(image_name), damage_rect),
                           extract_rectangle(restored_image, damage_rect))


def calculate_error(original, new):
    original = original.flatten()
    new = new.flatten()
    mse = np.sum((original - new) ** 2) / float(original.size)
    return mse


def compare_width_error(damage_rect, original_name, restored_name, params):
    widths = [1.0, 5.0, 10.0]
    for width in widths:
        params[0] = width
        error = test(damage_rect, f"../data/{original_name}.jpg", restored_name, *params)
        print(f"Error for width {width}: {error}")


def compare_radius_error(damage_rect, original_name, restored_name, params):
    radii = [20, 30, 40]
    for radius in radii:
        params[3] = radius
        error = test(damage_rect, f"../data/{original_name}.jpg", restored_name, *params)
        print(f"Error for radius {radius}: {error}")


if __name__ == '__main__':
    ideal_settings = [
        [10.0, 0.43, 1.0, 25, 3, cv.INPAINT_NS, (200, 350, 50, 50)],
        [8.0, 0.43, 1.0, 20, 3, cv.INPAINT_NS, (150, 160, 25, 25)],
        [5.0, 0.2, 1.0, 20, 3, cv.INPAINT_NS, (50, 150, 25, 25)],
        [12.0, 0.73, 2.0, 30, 10, cv.INPAINT_NS, (200, 100, 50, 50)]
    ]
    # compare_width_error((110, 10, 50, 50), "lion", "restored-lion",
    #                     ideal_settings[2])
    compare_radius_error((300, 185, 108, 108), "sand", "restored-sand",
                         ideal_settings[3])

    # test((300, 250, 100, 100), "../data/Granite-Rock.jpg", "restored-rocks", *ideal_settings[0])

    # test((150, 120, 48, 48), "../data/leaves-cropped.jpg", "restored-leaves", *ideal_settings[1])

    # test((110, 10, 50, 50), "../data/lion.jpg", "restored-lion", *ideal_settings[2])

    # test((300, 185, 108, 108), "../data/sand.jpg","restored-sand", *ideal_settings[3])
