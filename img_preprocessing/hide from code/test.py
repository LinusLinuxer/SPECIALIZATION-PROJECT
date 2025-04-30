# In general: This script is used in the whole programm to round it up.
# Our programm should be able to process images and prepare them for the model.
# The script is designed to work with images of different sizes and formats.


# This script contains functions to preprocess images for the model.
# It includes functions to load images, resize them, convert them to the appropriate format,
# and remove border noise.
# 1. load_image: Loads an image from a file path.
# 2. apply_gaussian_blur: Applies blur to reduce noise.
# 3. greyscale: Converts image to greyscale and applies Otsu's thresholding.
# 4. remove_border_noise: Detects the main content area and crops the image to remove borders.
# 5. resize_image: Resizes an image to the specified width and height.

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

# Constants for image processing (can be adjusted)
RESIZE_WIDTH = 400
RESIZE_HEIGHT = 200
BORDER_PADDING = 10  # Pixels to add around the detected content


# -------------------------------------------------------------------------------------#
class preprocess:
    def __init__(self, img_path):
        """Initialize the image processor with the image path"""
        self.img_path = img_path
        self.img = None  # Initialize img attribute

    def load_image(self):
        """Load the image from the specified path"""
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            logging.error(f"Image not found or could not be loaded at {self.img_path}")
            raise ValueError(f"Image not found at {self.img_path}")
        logging.info(f"Image loaded successfully: {self.get_image_name()}")
        return self.img

    def get_image_name(self):
        """Get the name of the image file without the path"""
        return os.path.basename(self.img_path)

    def apply_gaussian_blur(self):
        """Apply Gaussian blur to the image to reduce noise"""
        if self.img is None:
            logging.error("Image not loaded before apply_gaussian_blur.")
            raise ValueError("Image not loaded. Please call load_image() first.")
        # Apply Gaussian blur with a 7x7 kernel
        # SigmaX is calculated from kernel size if set to 0
        self.img = cv2.GaussianBlur(self.img, (7, 7), 0)
        logging.info("Applied Gaussian blur.")
        return self.img

    def greyscale(self):
        """Convert the image to greyscale and apply a binary threshold"""
        if self.img is None:
            logging.error("Image not loaded before greyscale.")
            raise ValueError("Image not loaded. Please call load_image() first.")
        # Convert image to grayscale
        grey_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        logging.info("Converted image to greyscale.")

        # Apply Otsu's thresholding to get a binary image
        # This automatically determines the optimal threshold value
        _, self.img = cv2.threshold(
            grey_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Note: THRESH_BINARY_INV makes the content white and background black,
        # which is often standard for contour detection.
        # If you need black text on white background, use THRESH_BINARY + THRESH_OTSU
        # and adjust the remove_border_noise logic accordingly.
        logging.info("Applied Otsu's thresholding.")
        return self.img

    def remove_border_noise(self, padding=BORDER_PADDING):
        """
        Removes border noise by finding the largest contour (assumed content)
        and cropping the image to its bounding box plus padding.
        Assumes the image has been thresholded (e.g., by greyscale method)
        with content being white (255) and background black (0).
        """
        if self.img is None:
            logging.error(
                "Image not processed (thresholded) before remove_border_noise."
            )
            raise ValueError("Image not thresholded. Please call greyscale() first.")

        if len(self.img.shape) != 2:
            logging.error("Image is not grayscale/binary for contour detection.")
            raise ValueError(
                "Image must be single-channel (binary/grayscale) for remove_border_noise."
            )

        # Find contours in the thresholded image.
        # cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
        # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments,
        # leaving only their end points.
        contours, _ = cv2.findContours(
            self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logging.warning("No contours found. Skipping border removal.")
            # Invert image back if needed (depending on desired output)
            self.img = cv2.bitwise_not(
                self.img
            )  # Uncomment if you want black text on white
            return self.img

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Get original image dimensions
        img_h, img_w = self.img.shape[:2]  # Use shape of the current self.img

        # Calculate padded coordinates, ensuring they stay within image bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        # Crop the image using the calculated bounding box coordinates
        # Important: Crop the version of the image you want to keep.
        # If you want the thresholded version cropped:
        cropped_img = self.img[y1:y2, x1:x2]
        # If you wanted to crop the original grayscale or even color image,
        # you'd need to load it again or store it before thresholding.
        # For this example, we crop the thresholded image.

        # Invert the image back to black text on white background if that's the desired final format
        # self.img = cv2.bitwise_not(cropped_img) # Uncomment if needed
        self.img = cropped_img  # Keep content white, background black

        logging.info(
            f"Removed border noise. Cropped to region: (x:{x1}, y:{y1}, w:{x2-x1}, h:{y2-y1})"
        )
        return self.img

    def resize_img(self, width=RESIZE_WIDTH, height=RESIZE_HEIGHT):
        """Resize the image to the specified width and height"""
        if self.img is None:
            logging.error("Image not loaded/processed before resize_img.")
            raise ValueError("Image not available. Please load/process it first.")
        resized_img = cv2.resize(
            self.img, (width, height), interpolation=cv2.INTER_AREA
        )
        logging.info(f"Resized image to {width}x{height}.")
        # Update self.img if you want the class instance to hold the resized image
        self.img = resized_img
        return self.img

    def show_img_matrix(self):
        """Display the image matrix (for debugging small images)"""
        if self.img is None:
            logging.error("Image not loaded before show_img_matrix.")
            raise ValueError("Image not loaded. Please call load_image() first.")
        print("Image matrix:")
        print(self.img)
        return self.img

    def show_img(self, window_name="Image"):
        """Display the image in a resizable window"""
        if self.img is None:
            logging.error("Image not loaded before show_img.")
            raise ValueError("Image not loaded. Please call load_image() first.")
        # Create a resizable window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Optional: Set a default size
        # cv2.resizeWindow(window_name, 800, 600)
        cv2.imshow(window_name, self.img)
        logging.info(f"Displaying image: {window_name}. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)  # Destroy only this specific window
        return self.img

    # --- Methods not yet implemented ---
    def split_img(self):
        # TODO: Implement image splitting logic
        logging.warning("split_img method is not implemented.")
        pass

    def create_folder(self):
        # TODO: Implement folder creation logic if needed for output
        logging.warning("create_folder method is not implemented.")
        pass

    def write_uml(self):
        # TODO: Implement XML/UML writing logic
        logging.warning("write_uml method is not implemented.")
        pass


# -------------------------------------------------------------------------------------#


def main():
    """Main processing loop"""
    # Get the directory of the current script
    try:
        script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    except NameError:
        # Handle case where script is run in an interactive environment
        script_path = os.getcwd()
        logging.info("Running in interactive mode, using current working directory.")

    # Get a list of all files in the directory
    try:
        filelist = os.listdir(script_path)
    except FileNotFoundError:
        logging.error(f"Directory not found: {script_path}")
        return

    logging.info(f"Scanning directory: {script_path}")
    processed_count = 0

    # Iterate over the files and check if they are images
    for file in filelist:
        # Simple check to avoid processing already processed files (adjust prefix/logic as needed)
        if (
            file.startswith("processed_")
            or file.startswith("grey_")
            or file.startswith("cropped_")
        ):
            logging.info(f"Skipping already processed file: {file}")
            continue

        # Check for common image file extensions
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
            img_path = os.path.join(script_path, file)
            file_name_base = os.path.splitext(file)[0]  # Get filename without extension
            logging.info(f"--- Processing image: {file} ---")

            try:
                # Create an instance of the preprocess class
                current_img = preprocess(img_path)

                # 1. Load the image
                current_img.load_image()
                # current_img.show_img("Original") # Optional: show original

                # 2. Apply gaussian blur (optional, helps with noise before thresholding)
                current_img.apply_gaussian_blur()
                # current_img.show_img("Blurred") # Optional: show blurred

                # 3. Convert to grayscale + thresholding (content becomes white)
                current_img.greyscale()
                # current_img.show_img("Thresholded (Content White)") # Optional: show thresholded

                # 4. Remove border noise by cropping to largest contour
                current_img.remove_border_noise(padding=BORDER_PADDING)
                # current_img.show_img("Border Removed") # Optional: show after border removal

                # 5. Resize the image (optional)
                # current_img.resize_img(RESIZE_WIDTH, RESIZE_HEIGHT)
                # current_img.show_img("Resized") # Optional: show resized

                # --- Save the processed image ---
                if current_img.img is not None and current_img.img.size > 0:
                    # Create a filename for the processed image
                    # timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # filename = f"processed_{file_name_base}_{timestamp}.png" # Save as PNG to avoid JPEG compression artifacts
                    filename = f"processed_{file_name_base}.png"  # Simpler filename
                    save_path = os.path.join(script_path, filename)

                    # Invert final image if you want black text on white background for saving
                    # final_image_to_save = cv2.bitwise_not(current_img.img)
                    final_image_to_save = (
                        current_img.img
                    )  # Save as is (white text on black bg)

                    if cv2.imwrite(save_path, final_image_to_save):
                        logging.info(
                            f"Successfully saved processed image to: {save_path}"
                        )
                        processed_count += 1
                    else:
                        logging.error(f"Failed to save processed image: {save_path}")

                else:
                    logging.warning(
                        f"Image processing resulted in an empty image for: {file}. Skipping save."
                    )

            except ValueError as e:
                logging.error(f"Value error processing {file}: {e}")
            except cv2.error as e:
                logging.error(f"OpenCV error processing {file}: {e}")
            except Exception as e:
                # Catch any other unexpected errors during processing of a single file
                logging.error(
                    f"Unexpected error processing {file}: {e}", exc_info=True
                )  # Log traceback

        else:
            # Optional: Log files that are skipped
            # logging.debug(f"Skipping non-image file: {file}")
            pass

    logging.info(f"--- Processing complete. Processed {processed_count} images. ---")


# --- Script execution ---
if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------#
