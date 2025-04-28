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

    def find_main_content_bounding_box(self):
        """
        Find the bounding box of the main content (largest connected component).
        Assumes the image is binary (black text on white background).
        Returns the coordinates of the bounding box as (x, y, w, h).
        """
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before find_main_content_bounding_box()."
            )
        # Check if the image is binary (0s and 255s)
        if not np.array_equal(np.unique(self.img), [0, 255]):
            raise ValueError("Image is not binary. Please convert it to binary first.")

        logging.info(
            "Finding the bounding box of the main content (largest component)..."
        )

        # Find contours in the binary image
        contours, _ = cv2.findContours(
            self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No contours found in the image.")

        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_box = (x, y, w, h)
        logging.info(f"Found bounding box of main content: {bounding_box}")

        height, width = self.img.shape
        logging.info(f"Image dimensions: {height} x {width}")
        # Optionally, you can draw the bounding box on the image for visualization
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the image to the bounding box
        self.img = self.img[y : y + h, x : x + w]
        # Optionally, you can draw the bounding box on the image for visualization
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        logging.info(f"Cropped image to bounding box: x={x}, y={y}, w={w}, h={h}")

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

        # find the bounding box of the main content and crop the image
        croppend_img = current_img.find_main_content_bounding_box()

        # save the cropped (and greyscaled) image
        if current_img is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"cropped_image_{timestamp}.jpeg"
            cv2.imwrite(filename, croppend_img)
        else:
            logging.warning(f"Could not crop image: {file}. Skipping save.")
    else:
        continue

# -------------------------------------------------------------------------------------#
