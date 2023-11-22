# from src.transfer import texture_transfer
import matlab.engine
import cv2 as cv
import os

def generate_texture_transfer(input_texture, target_image):
    save_image_relative_path(input_texture, "./data/texture.jpg")
    save_image_relative_path(target_image, "./data/target.jpg")
    eng = matlab.engine.start_matlab()
    eng.run("./src/transfer_script/transfer.m", nargout=0)
    eng.quit()

def save_image_relative_path(image, relative_path):
    """
    Save an image using a relative path.

    Parameters:
    - image: The image array to be saved.
    - relative_path: The relative path where the image will be saved.
    """
    # Get the absolute path based on the current working directory
    absolute_path = os.path.abspath(relative_path)

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    # Save the image
    a = cv.imwrite(absolute_path, image)

if __name__ == "__main__":
    generate_texture_transfer(None, None)
