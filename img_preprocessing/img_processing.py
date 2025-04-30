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

import time
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants for image processing
WIDTH = 400
HEIGHT = 200


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

    def get_image_name(self):
        """Get the name of the image file without the path"""
        return os.path.basename(self.img_path)

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

        # logging.info("Applying Gaussian blur...")
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
            self.img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        return self.img

    def dilate_img(self):
        """Dilate the image to enhance the features"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before dilate_img()."
            )
        # dilute the image to enhance the features, (especially sideways)
        kernel = np.ones((1, 40), np.uint8)
        self.img = cv2.dilate(self.img, kernel, iterations=1)
        logging.info("Dilating the image...")
        return self.img

    def line_segmentation(self):
        """Segment the image into lines using contours"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before line_segmentation()."
            )

        # Find contours in the image
        contours, hierarchy = cv2.findContours(
            self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by their y-coordinate (top to bottom)
        sorted_contours_lines = sorted(
            contours, key=lambda ctr: cv2.boundingRect(ctr)[1]
        )

        # Draw contours on the image
        img_with_contours = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for ctr in sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (41, 55, 214), 10)
        self.img = img_with_contours

        logging.info(f"Number of contours found: {len(sorted_contours_lines)}")

        return self.img

    def resize_img(self, width, height):
        """Resize the image to the specified width and height"""
        resized_img = cv2.resize(self.img, (width, height))
        return resized_img

    def show_img_matrix(self):
        """Display the image matrix"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before show_img_matrix()."
            )
        # Display the image matrix
        print("Image matrix:")
        print(self.img)
        return self.img

    def show_img(self):
        """Display the image"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before show_img()."
            )
        # Display the image
        # set window size
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.img


# -------------------------------------------------------------------------------------#

# get the path of the current script
path = os.path.dirname(os.path.abspath(sys.argv[0]))

# get a list of all files in the directory
filelist = os.listdir(path)
# print(f"path: {path}\nFiles in the directory: {filelist}")

# iterate over the files and check if they are images
for file in filelist:
    # check if the file already has been greyscaled
    if file.startswith("grey_") or file.startswith("cropped"):
        continue

    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
        # load the image
        img_path = os.path.join(path, file)
        file_name = os.path.basename(img_path)
        logging.info(f"Processing image: {file_name}")

        # create an instance of the preprocess class
        current_img = preprocess(img_path)

        # load the image
        current_img.load_image()

        # apply gaussian blur (on RGB)
        current_img.apply_gaussian_blur()

        # convert to grayscale + thresholding
        current_img.greyscale()
        current_img.dilate_img()
        current_img.line_segmentation()

        # save the cropped (and greyscaled) image
        if current_img is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{file_name}_{timestamp}.jpeg"
            cv2.imwrite(filename, current_img.img)
        else:
            logging.warning(f"Could not crop image: {file}. Skipping save.")
    else:
        continue

# -------------------------------------------------------------------------------------#

current_img.show_img()
