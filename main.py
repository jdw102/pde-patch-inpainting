import cv2 as cv
from code.image_util import load_damaged_image, extract_rectangle
from code.pde_patch_pipeline import pde_patch




def test(damage_rect, texture_rectangle, image_name):
    damaged_image, mask = load_damaged_image(damage_rect, image_name)
    restored_image = pde_patch(damaged_image, mask, damage_rect, texture_rectangle)
    # Display the original, damaged, and restored images
    cv.imshow("Original Image", damaged_image)
    cv.imshow("Restored Image", restored_image)
    # Wait for a key press and then close the windows
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # test((120, 160, 10, 10), "./data/abbeyroad.jpg")
    # test((200, 160, 50, 50), "./data/cat_drawing.jpg")
    test((290, 250, 100, 100), (260, 280, 30, 30),"./data/Granite-Rock.jpg")
    # test((290, 250, 100, 100), "./data/camo_hand.jpg")

