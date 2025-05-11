import os
import cv2


def apply_gaussian_blur(image_path, output_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unable to read")
        blurred_image = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.imwrite(output_path, blurred_image)
        print(f"Blurred image saved to {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, f"{filename}")
            apply_gaussian_blur(input_path, output_path)


if __name__ == "__main__":
    current_directory = os.getcwd()
    process_images_in_directory(current_directory)
