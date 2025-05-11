# Engelbrecht | Nadarajah 2025

"""
This script is designed for preprocessing and segmenting images, particularly for tasks involving line segmentation.
It provides a class `preprocess` that includes various methods to load, process, and manipulate images.
The script performs operations such as cropping, Gaussian blurring, grayscale conversion, thresholding,
dilation, line segmentation, and saving segmented lines as separate images.

The main functionality iterates through all image files in the current directory, processes them,
and saves the results in a structured output folder. This is useful for preparing images for further
analysis or machine learning tasks.
"""


import os
import sys
import cv2
import numpy as np
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants for image processing
WIDTH = 400
HEIGHT = 200
x_kernel_size = 50  # Dilution kernel size


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

    def create_folder(self, path):
        """Create a folder to save the rows of the processed images"""
        folder_name = self.get_image_name()
        new_folder_path = os.path.join(path, folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path, exist_ok=True)
            # logging.info(f"Folder created: {new_folder_path}")
        return new_folder_path

    def crop_img(self):
        """Crop the image by 5% from every edge"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before crop_img()."
            )
        height, width = self.img.shape[:2]
        # Crop 5% from each edge
        crop_x = int(width * 0.05)
        crop_y = int(height * 0.05)
        self.img = self.img[crop_y : height - crop_y, crop_x : width - crop_x]
        return self.img

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

    def greyscale(self):
        """Convert the image to greyscale and apply a binary threshold"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before greyscale()."
            )
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # logging.info("Converting to greyscale...")

        # Apply a binary threshold to convert the greyscale image to black and white
        _, self.img = cv2.threshold(
            self.img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        return self.img

    def dilate_img(self, x_kernel_size: int):
        """Dilate the image to enhance the features"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before dilate_img()."
            )
        # dilute the image to enhance the features, (especially sideways)
        kernel = np.ones((1, x_kernel_size), np.uint8)
        self.img = cv2.dilate(self.img, kernel, iterations=1)
        # logging.info("Dilating the image...")
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
        self.sorted_contours_lines = sorted(
            contours, key=lambda ctr: cv2.boundingRect(ctr)[1]
        )

        # logging.info(f"Number of contours found: {len(self.sorted_contours_lines)}")

        # Return the processed image and the number of contours
        return self.img, len(self.sorted_contours_lines)

    def remove_bad_segementaion(self):
        """Remove bad segmentations based on size"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before remove_bad_segementaion()."
            )

        # Get the height and width of the image
        height, width = self.img.shape[:2]
        min_line_width = int(width * 0.35)
        min_line_height = int(height * 0.01)

        # Filter out bad contours
        filtered_contours = [
            ctr
            for ctr in self.sorted_contours_lines
            if cv2.boundingRect(ctr)[2] >= min_line_width
            and cv2.boundingRect(ctr)[3] >= min_line_height
        ]

        # filter out contours that are taller in height than width
        filtered_contours = [
            ctr
            for ctr in filtered_contours
            if cv2.boundingRect(ctr)[3] <= cv2.boundingRect(ctr)[2]
        ]

        # logging.info(
        #     f"Number of contours before filtering: {len(self.sorted_contours_lines)}"
        # )
        # logging.info(f"Number of contours after filtering: {len(filtered_contours)}")

        # Update the sorted_contours_lines with the filtered contours
        self.sorted_contours_lines = filtered_contours
        return self.sorted_contours_lines

    def draw_segmentation(self):
        """Draw the segmented lines on the image"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before draw_segmentation()."
            )

        # Ensure the image is in BGR format
        if len(self.img.shape) == 2:  # Grayscale image
            img_with_lines = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        else:  # Already a color image
            img_with_lines = self.img.copy()

        # Draw the segmented lines on the image
        for ctr in self.sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(
                img_with_lines, (x, y), (x + w, y + h), (125, 255, 125), 10
            )  # Green color for contours
        self.img = img_with_lines
        return self.img

    def split_img(self, folder_path, sorted_contours_lines):
        """Cut the image into smaller images based on the segmented lines"""
        if self.img is None:
            raise ValueError(
                "Image not loaded. Please call load_image() before cut_img()."
            )
        # Ensure the folder path exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        os.chdir(folder_path)
        # Iterate over the sorted contours and save each segmented line as a separate image
        for i, ctr in enumerate(sorted_contours_lines):
            x, y, w, h = cv2.boundingRect(ctr)
            # Crop the image using the bounding rectangle
            cropped_img = self.img[y : y + h, x : x + w]

            # Save the cropped image with a unique filename
            filename = f"line_{i}.png"
            cv2.imwrite(filename, cropped_img)
            # logging.info(f"Saved segmented line {i} as {filename}")

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

# iterate over the files and check if they are images
progress_bar = tqdm(filelist, desc="Processing images", unit="file")
for file in progress_bar:
    # Update the progress bar description with the current file being processed
    progress_bar.set_description(f"Processing {file}")
    # check if the file is an image
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
        # load the image
        img_path = os.path.join(path, file)
        file_name = os.path.basename(img_path)
        # logging.info(f"Processing image: {file_name}")

        # create an instance of the preprocess class
        current_img = preprocess(img_path)

        # load the image
        current_img.load_image()
        current_img.crop_img()

        # apply gaussian blur (on RGB)
        current_img.apply_gaussian_blur()

        # convert to grayscale + thresholding
        current_img.greyscale()
        current_img.dilate_img(x_kernel_size)
        current_img.line_segmentation()
        current_img.remove_bad_segementaion()
        current_img.draw_segmentation()
        # current_img.show_img()

        # save the cropped (and greyscaled) image
        if current_img is not None:
            # Create a new filename for the processed image
            filename = f"{file_name}"
            output_folder = os.path.join(path, "processed_images")
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, filename)

            # create output folder for split images
            split_folder = os.path.join(output_folder, current_img.get_image_name())
            os.makedirs(split_folder, exist_ok=True)

            # Save the image to the created split folder
            split_output_path = os.path.join(split_folder, filename)
            cv2.imwrite(split_output_path, current_img.img)
            # logging.info(f"Image saved as: {split_output_path}")

            # copy the original image to the output folder
            original_img = cv2.imread(img_path)
            cv2.imwrite(output_path, original_img)

            # copy the original image to the split folder
            current_img.img = original_img
            current_img.greyscale()

            # split the image into rows
            # use the sorted contours to split the image into rows
            current_img.split_img(split_folder, current_img.sorted_contours_lines)

            filelist_split_folder = os.listdir(split_folder)
            # iterate over the files in the split folder

            # get the dimensions of the image
            _, width = current_img.img.shape[:2]

            for file in filelist_split_folder:
                # ignore the original image
                if file == current_img.get_image_name():
                    continue
                # Read the file as an image
                file_path = os.path.join(split_folder, file)
                image = cv2.imread(file_path)

                # Ensure the image was loaded successfully
                if image is None:
                    logging.warning(f"Could not read image: {file}. Skipping resize.")
                    continue

                # Get the height of the current image
                height, _ = image.shape[:2]

                # Resize the image to the same width as the original
                resized_image = cv2.resize(
                    image,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Save the resized image back to the split folder
                cv2.imwrite(file_path, resized_image)

        else:
            logging.warning(f"Could not crop image: {file}. Skipping save.")
    else:
        continue
