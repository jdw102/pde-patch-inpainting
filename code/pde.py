import numpy as np
import cv2 as cv


def test(damage_rect, image_name):
    image = cv.imread(image_name)
    # Create a mask to represent the damaged area
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv.rectangle(mask, (damage_rect[0], damage_rect[1]),
                 (damage_rect[0] + damage_rect[2], damage_rect[1] + damage_rect[3]), 255, -1)
    # Damage the image by turning the patch black
    image[damage_rect[1]:damage_rect[1] + damage_rect[3], damage_rect[0]:damage_rect[0] + damage_rect[2]] = 0
    # Use the inpainting function to restore the damaged area
    restored_image = cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
    # Display the original, damaged, and restored images
    cv.imshow("Original Image", image)
    cv.imshow("Damaged Image", mask)
    cv.imshow("Restored Image", restored_image)
    # Wait for a key press and then close the windows
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    # test((120, 160, 10, 10), "../data/abbeyroad.jpg")
    # test((200, 160, 50, 50), "../data/cat_drawing.jpg")
    # test((290, 250, 100, 100), "../data/rocks.jpg")
    # test((290, 250, 100, 100), "../data/camo_hand.jpg")