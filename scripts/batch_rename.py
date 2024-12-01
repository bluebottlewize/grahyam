import os
import re
import shutil

COPY_FROM = "../dump/mal-htr/"
COPY_TO = "../test/"

def get_last_number(folder_path):
    files = os.listdir(folder_path)
    pattern = re.compile(r'(\d+).txt$')

    max_number = -1

    for file in files:
        match = pattern.search(file)
        if match:
            number = int(match.group(1))  # Extract the number part and convert to int
            if number > max_number:
                max_number = number

    if max_number != -1:
        # print(f"The last number is: {max_number:06d}")  # Print with leading zeros (6 digits)
        return max_number
    else:
        return 0

print(get_last_number('../test/ക/'))


def copy_files(src, dest, letter):
    pattern = re.compile(r'(\d+).txt$')
    print(src, dest, letter)

    os.makedirs(dest, exist_ok=True)

    last_file_in_dest = get_last_number(dest)
    print(f"last file in {dest}", last_file_in_dest)

    for root, dirs, files in os.walk(src):
        for file in files:
            match = pattern.search(file)
            if match:
                number = int(match.group(1)) + last_file_in_dest
                formatted_filename = f"{letter}_{number:06d}.txt"

                src_path = os.path.join(src, file)
                dest_path = os.path.join(dest, formatted_filename)

                print('copying', src_path, 'to', dest_path)
                shutil.copy(src_path, dest_path)


def iterate_directory(parent_folder):
    for root, dirs, files in os.walk(parent_folder):

        dir_path = ""
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            print(f"Found directory: {dir_path}", get_last_number(dir_path))
            dest_dir_path = os.path.join(COPY_TO, dir_name)
            copy_files(dir_path, dest_dir_path, dir_name)


iterate_directory(COPY_FROM)
