import cv2 as cv


def pde_inpaint(image, mask, radius):
    return cv.inpaint(image, mask, inpaintRadius=radius, flags=cv.INPAINT_NS)
