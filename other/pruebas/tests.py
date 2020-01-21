from PIL import Image
from PIL import ImageFilter
import os
import glob


def change_extension(folder_path, current_extension, desired_extension):
    paths = folder_path + r'\*' + current_extension
    paths = glob.glob(paths)

    for path in paths:
        no_ext, _ = os.path.splitext(path)
        os.rename(path, no_ext + desired_extension)


if __name__ == '__main__':
    folder_path = r'C:\Users\Ceiec06\Documents\GitHub\CEIEC-GANs\greek_temples_dataset\restored_png'
    change_extension(folder_path, '.jpg', '.png')
