import os

labels = []

def iterate_directory(parent_folder):
    for root, dirs, files in os.walk(parent_folder):

        dir_path = ""
        for dir_name in dirs:
            labels.append(dir_name)

iterate_directory('../train/')

sorted_list = sorted(labels, key=lambda x: (len(x), x))

print(sorted_list)
