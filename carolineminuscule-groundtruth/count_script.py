# This script counts the number of top-level directories, text files, and image files in the current directory.

import os


def count_top_directories_and_files(folder_path):
    txt_files = 0
    png_files = 0
    xml_files = 0
    folder_count = 0

    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        # Check if the entry is a directory or a file
        # If it's a directory, recursively count its contents
        if os.path.isdir(entry_path):
            sub_txt_files, sub_png_files, sub_xml_files = (
                count_top_directories_and_files(entry_path)
            )

            folder_count += 1
            txt_files += sub_txt_files
            png_files += sub_png_files
            xml_files += sub_xml_files
        # If it's a file, check its extension and increment the appropriate counter
        elif os.path.isfile(entry_path):
            if entry.endswith(".txt"):
                txt_files += 1
            elif entry.endswith(".png"):
                png_files += 1
            elif entry.endswith(".xml"):
                xml_files += 1

    return txt_files, png_files, xml_files


def count_png_in_subfolders(folder_path):
    """Calculate the average number of PNG files in subfolders"""
    png_counts = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            _, sub_png_files, _ = count_top_directories_and_files(entry_path)
            png_counts.append(sub_png_files)
    return sum(png_counts) / len(png_counts) if png_counts else 0


# Get the current directory of the script
folder_path = os.path.dirname(os.path.abspath(__file__))
print(f"Counting files in: {folder_path}")

txt_files, png_files, xml_files = count_top_directories_and_files(folder_path)

print(f"Text files (.txt): {txt_files}")
print(f"Image files (.png): {png_files}")
print(f"Image files (.xml): {xml_files}")
print(f"Total files: {txt_files + png_files + xml_files}")
# every directory should have at least one xml file
print(
    f"The average number of files per directory: {int((txt_files + png_files + xml_files) / xml_files if xml_files > 0 else 0)}"
)

# Calculate the average number of PNG files in subfolders
average_png_per_subfolder = count_png_in_subfolders(folder_path)
print(f"The average number of PNG files in subfolders: {average_png_per_subfolder}")
