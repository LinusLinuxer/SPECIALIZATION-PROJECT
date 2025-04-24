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

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants for image processing
WIDTH = 400
HEIGHT = 200


#! Remove this line after devolopment
# this is a test to remove the file
# remove the file if it exists
if os.path.exists("grey_bsb00046285.0011.jpeg"):
    os.remove("grey_bsb00046285.0011.jpeg")
else:
    print("The file does not exist")


#! End of remove line
# -------------------------------------------------------------------------------------#
class preprocess:
    def __init__(self, img_path):
        """Initialize the image processor with the image path"""
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
        # look here for the parameters:
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        self.img = cv2.GaussianBlur(self.img, (7, 7), 0)

        logging.info("Applying Gaussian blur...")
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
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        logging.info("Converting to greyscale...")

        # Apply a binary threshold to convert the greyscale image to black and white
        _, self.img = cv2.threshold(
            self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return self.img

    def remove_marginal_noise(image, margin=10):
        """
        Trims a fixed margin from the image edges.
        You can combine this with thresholding or edge detection to be more dynamic.
        """
        height, width = image.shape[:2]

        return image[margin : height - margin, margin : width - margin]

    def crop_whitespace(self):
        """
        Crops the white borders from a binary or grayscale image, ignoring small noise.
        Assumes background is white (255).
        """
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before crop_whitespace()."
            )

        # Convert to grayscale if the image is in color
        # shape is the amount of channels
        # if the image is in color, it has 3 channels (RGB)
        if len(self.img.shape) == 3:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Threshold to get a binary image
        _, binary = cv2.threshold(self.img, 240, 255, cv2.THRESH_BINARY)

        # Invert the binary image
        inverted = cv2.bitwise_not(binary)

        # Find contours of the non-white regions
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Filter out small contours based on area to ignore noise
            min_area = 1000  # Minimum area threshold for contours
            filtered_contours = [
                contour for contour in contours if cv2.contourArea(contour) > min_area
            ]
            if filtered_contours:
                # Get the bounding box of the largest filtered contour
                x, y, w, h = cv2.boundingRect(filtered_contours[0])

                # Crop the image using the bounding box
                self.img = self.img[y : y + h, x : x + w]
                logging.info("Cropping whitespace while ignoring noise...")
            else:
                logging.warning(
                    "No significant contours found. Image might be noisy or empty."
                )

        return self.img

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
    if file.startswith("grey_") or file.startswith("grey_") or file.startswith("grey_"):
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
        current_img.greyscale()

        # crop whitespace
        # current_img.crop_whitespace()
        current_img.crop_whitespace()

        # save the image
        cv2.imwrite(os.path.join(path, "grey_" + file), current_img.img)
    else:
        continue

# -------------------------------------------------------------------------------------#

# Display the processed greyscale image with a set window size
cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)  # Allow window resizing
cv2.resizeWindow("Processed Image", 800, 600)  # Set the window size to 800x600
cv2.imshow("Processed Image", current_img.img)  # Show the image in the window
# Wait for a key press and then close the window
cv2.waitKey(0)
# Close the image window
cv2.destroyAllWindows()
