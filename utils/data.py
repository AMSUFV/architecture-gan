# TODO: add a 'Deprecated' folder to store the unused scripts once refactoring is done
import glob
import os
import tensorflow as tf
from functools import reduce
from utils import preprocessing

path_dataset = 'dataset'
path_temples = '/temples'
path_temples_ruins = '/temples_ruins'
path_temples_colors = '/colors_temples'
path_temples_ruins_colors = '/colors_temples_ruins'

repetitions = [1, 2]

mapping_func = preprocessing.load_images
glob_pattern = '/*temple_{}*/*'


def get_dataset(path, option, *args):

    option = option.lower()
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if option == 'reconstruction':
        x_path = path + path_temples_ruins
        y_path = path + path_temples
        return reconstruction(*args, x_path, y_path)

    elif option == 'color_reconstruction':
        x_path = path + path_temples_ruins_colors
        y_path = path + path_temples_colors
        return reconstruction(*args, x_path, y_path)

    elif option == 'color_assisted':
        x_path = path + path_temples_ruins
        y_path = path + path_temples_colors
        z_path = path + path_temples
        return reconstruction(*args, x_path, y_path, z_path)

    elif option in ['masking', 'de-masking']:
        preprocessing.apply_mask = True
        x_path = path + path_temples_ruins_colors
        y_path = path + path_temples_colors
        z_path = path + path_temples
        if option == 'de-masking':
            preprocessing.demasking = True
            return reconstruction(*args, x_path, y_path, z_path)
        if option == 'masking':
            aux_path = path + path_temples_ruins
            return reconstruction(*args, x_path, y_path, z_path, aux_path)

    elif option == 'segmentation':
        x_path = [path + path_temples_ruins, path + path_temples]
        y_path = [path + path_temples_ruins_colors, path + path_temples_colors]
        return reconstruction(*args, x_path, y_path)

    elif option == 'de-segmentation':
        x_path = [path + path_temples_ruins_colors, path + path_temples_colors]
        y_path = [path + path_temples_ruins, path + path_temples]
        return reconstruction(*args, x_path, y_path)

    else:
        raise Exception('Option not supported. Run train.py -h to see the supported options.')


def reconstruction(temples, split=0.25, batch_size=1, buffer_size=400, *paths):
    files = list(map(lambda x: simple(x, paths), temples))
    files = reduce(concat, files)

    # dataset size
    size = list(map(lambda x: len(glob.glob(paths[0] + glob_pattern.format(x))), temples))
    size = reduce((lambda x, y: x + y), size)

    train_files, val_files = train_val_split(files, split, size, buffer_size)

    train = train_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)
    val = val_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)

    return train, val


def simple(number, paths):
    pattern = glob_pattern.format(number)
    if type(paths[0]) == list:  # in case several glob patterns are needed
        paths = [[path + pattern for path in path_list] for path_list in paths]
    else:
        paths = [path + pattern for path in paths]
    file_datasets = [tf.data.Dataset.list_files(path, shuffle=False).repeat(rep)
                     for path, rep in zip(paths, repetitions)]
    return tf.data.Dataset.zip(tuple(file_datasets))


def concat(a, b):
    return a.concatenate(b)


def train_val_split(dataset, split, size, buffer_size):
    dataset = dataset.shuffle(buffer_size)
    train = dataset.skip(round(size * split))
    val = dataset.take(round(size * split))
    return train, val
