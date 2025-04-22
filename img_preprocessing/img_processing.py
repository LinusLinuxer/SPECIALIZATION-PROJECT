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

# pip install opencv-python


class preprocess:
    def __init__(self, path, img):
        self.path = path
        self.img = img

    def split_img(self):
        pass

    def greyscale(self):
        # Convert the image to grayscale
        gray_scaled = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return gray_scaled


# get a list of all files in the directory
path = os.path.dirname(os.path.abspath(sys.argv[0]))
filelist = os.listdir(path)
print(f"path: {path}, Files in the directory: {filelist}")

i = 0
# iterate over the files and check if they are images
for i in filelist[i]:
    # img_path = path.join(path, file)
    if i.endswith(".jpg") or i.endswith(".png"):
        # load the image
        img = cv2.imread(i)
        # create an instance of the preprocess class
        current_img = preprocess(path, img)
        # convert the image to grayscale
        gray_scaled = current_img.greyscale()
        # save the image
        cv2.imwrite(os.path.join(path, "gray_" + i), gray_scaled)
    else:
        continue
