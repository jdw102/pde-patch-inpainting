import cv2 as cv
from src.image_util import load_damaged_image, extract_rectangle, load_image
from src.pde_patch_pipeline import pde_patch
from src.texture_transfer import generate_texture_transfer


def test(damage_rect, texture_rectangle, image_name):
    damaged_image, mask = load_damaged_image(damage_rect, image_name)
    restored_image = pde_patch(damaged_image, mask, damage_rect, texture_rectangle)
    # Texture transfer and synthesis
    input_texture = extract_rectangle(damaged_image, (150, 380, 200, 200))
    cv.imwrite("./data/texture.jpg", input_texture)
    input_patch = extract_rectangle(restored_image, damage_rect)
    cv.imwrite("./data/restored.jpg", input_patch)
    generate_texture_transfer(input_texture, restored_image)
    # Display the original, damaged, and restored images
    cv.imshow("Original Image", damaged_image)
    cv.imshow("Restored Image", restored_image)
    cv.imshow("Restored Patch", input_patch)
    cv.imshow("Input Texture", input_texture)
    # generate_texture_transfer("./data/rice.jpg")
    # Wait for a key press and then close the windows
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # test((120, 160, 10, 10), "./data/abbeyroad.jpg")
    # test((200, 160, 50, 50), "./data/cat_drawing.jpg")
    test((290, 250, 100, 100), (260, 280, 30, 30), "./data/Granite-Rock.jpg")
    # test((290, 250, 100, 100), "./data/camo_hand.jpg")
