# Engelbrecht | Nadarajah 2025

# In general: This script is used in the whole programm to round it up.
# Our programm should be able to process images and prepare them for the model.
# The script is designed to work with images of different sizes and formats.


# This script contains functions to preprocess images for the model.
# It includes functions to load images, resize them, and convert them to the appropriate format.
# 1. load_image: Loads an image from a file path.
# 2. cut images into patches: Cuts an image into smaller patches.
# 3. resize_image: Resizes an image to the specified width and height.

# use maseked grey scale images, these, according to "Cross-codex Learning for Reliable Large Scale
# Scribe Identi cation in Medieval Manuscripts" provide the best results


import os
import sys
import cv2
import numpy as np


# Constants for image processing
WIDTH = 400
HEIGHT = 200


#! Remove this line after devolopment
if os.path.exists("gray_bsb00046285.0011.jpeg"):
    os.remove("gray_bsb00046285.0011.jpeg")
else:
    print("The file does not exist")


#! End of remove line
# -------------------------------------------------------------------------------------#
class preprocess:
    def __init__(self, img_path):
        # Load the image from the specified path
        self.img_path = img_path

    def load_image(self):
        """Load the image from the specified path"""
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise ValueError(f"Image not found at {self.img_path}")
        return self.img

    def split_img(self):
        # TODO: Split the image into rows
        # algorithm idea:
        # find line with the most avg. black pixels
        # thicken the line until a certain threshold is reached (more white than black pixels)
        # follow black pixel to include ascender and descender.
        pass

    def create_folder(self):
        pass

    def apply_gaussian_blur(self):
        """Apply Gaussian blur to the image to reduce noise"""

        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before apply_gaussian_blur()."
            )
        #   use cv2.GaussianBlur()
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        return self.img

    def write_uml(self):
        # Todo: This might be the hardest part and also depending on how you approach it,
        #   optional. We dont need no XML file, but it would be really nice to have.
        #   write height and width of file
        #   filename
        #   write the polygons of the uml-file
        pass

    def greyscale(self):
        """Convert the image to greyscale and apply a binary threshold"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before greyscale()."
            )
        grey_scaled = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to convert the greyscale image to black and white
        _, grey_scaled = cv2.threshold(
            grey_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return grey_scaled

    def set_page_frame(self):
        """Set the page frame to the image.
        Remove the white space around the image.
        """
        pass

    def resize_img(self, width, height):
        """Resize the image to the specified width and height"""
        resized_img = cv2.resize(self.img, (width, height))
        return resized_img


# -------------------------------------------------------------------------------------#

# get the path of the current script
path = os.path.dirname(os.path.abspath(sys.argv[0]))

# get a list of all files in the directory
filelist = os.listdir(path)
# print(f"path: {path}\nFiles in the directory: {filelist}")

# iterate over the files and check if they are images
for file in filelist:
    # check if the file already has been greyscaled
    if file.startswith("gray_") or file.startswith("gray_") or file.startswith("gray_"):
        continue

    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
        # load the image
        img_path = os.path.join(path, file)

        # create an instance of the preprocess class
        current_img = preprocess(img_path)

        # load the image
        current_img.load_image()

        # apply gaussian blur (on RGB)
        current_img.apply_gaussian_blur()

        # convert to grayscale + thresholding
        grey_scaled = current_img.greyscale()

        # save the image
        cv2.imwrite(os.path.join(path, "gray_" + file), grey_scaled)
    else:
        continue

# -------------------------------------------------------------------------------------#

# Display the processed greyscale image with a set window size
cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)  # Allow window resizing
cv2.resizeWindow("Processed Image", 800, 600)  # Set the window size to 800x600
cv2.imshow("Processed Image", grey_scaled)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()  # Close the image window
