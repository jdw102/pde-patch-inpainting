import cv2 as cv
from src.image_util import load_damaged_image, extract_rectangle, load_image
from src.pde_patch_pipeline import pde_patch
from src.texture_transfer import generate_texture_transfer


def test(damage_rect, texture_rectangle, image_name):
    damaged_image, mask = load_damaged_image(damage_rect, image_name)
    restored_image = pde_patch(damaged_image, mask, damage_rect, texture_rectangle)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # test((120, 160, 10, 10), "./data/abbeyroad.jpg")
    # test((200, 160, 50, 50), "./data/cat_drawing.jpg")
    test((290, 250, 100, 100), (100, 400, 100, 200), "./data/Granite-Rock.jpg")
    # test((290, 250, 100, 100), "./data/camo_hand.jpg")
