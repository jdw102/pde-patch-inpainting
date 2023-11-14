import cv2 as cv
from code.image_util import load_damaged_image, extract_rectangle
from code.pde import pde_inpaint


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def test(damage_rect, image_name):
    damaged_image, mask = load_damaged_image(damage_rect, image_name)
    restored_image = pde_inpaint(damaged_image, mask, 3)
    restored_patch = extract_rectangle(restored_image, damage_rect)
    # Display the original, damaged, and restored images
    cv.imshow("Original Image", damaged_image)
    cv.imshow("Damaged Image", mask)
    cv.imshow("Restored Image", restored_image)
    cv.imshow("Restored Patch", restored_patch)
    # Wait for a key press and then close the windows
    cv.waitKey(0)
    cv.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test((120, 160, 10, 10), "./data/abbeyroad.jpg")
    # test((200, 160, 50, 50), "./data/cat_drawing.jpg")
    test((290, 250, 100, 100), "./data/Granite-Rock.jpg")
    # test((290, 250, 100, 100), "./data/camo_hand.jpg")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
