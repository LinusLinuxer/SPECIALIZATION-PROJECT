# Engelbrecht | Nadarajah 2025

# In general: This script is used in the whole programm to round it.
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

WIDTH = 400
HEIGHT = 200


class preprocess:
    def __init__(self, img_path):
        # Load the image from the specified path
        self.img_path = img_path

    def load_image(self):
        # Load the image from the specified path
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise ValueError(f"Image not found at {self.img_path}")
        return self.img

    def split_img(self):
        # TODO: Split the image into rows
        pass

    def write_uml(self):
        # Todo: This might be the hardest part
        # write the polygons of the uml-file
        pass

    def greyscale(self):
        # Convert the image to grayscale
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before greyscale()."
            )
        gray_scaled = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to convert the grayscale image to black and white
        _, gray_scaled = cv2.threshold(gray_scaled, 127, 255, cv2.THRESH_BINARY)
        return gray_scaled

    def resize_img(self, width, height):
        # Resize the image to the specified width and height
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
    # check if the file already has been grayscaled
    if (
        file.startswith("gray.jpg")
        or file.startswith("gray.png")
        or file.startswith("gray.jpeg")
    ):
        continue

    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
        # load the image
        img_path = os.path.join(path, file)
        # create an instance of the preprocess class
        current_img = preprocess(img_path)
        # load the image
        current_img.load_image()
        # convert the image to grayscale
        gray_scaled = current_img.greyscale()
        # save the image
        cv2.imwrite(os.path.join(path, "gray_" + file), gray_scaled)
    else:
        continue
