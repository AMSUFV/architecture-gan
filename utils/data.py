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
len_dataset = 100

simple_options = ['reconstruction', 'color_reconstruction', 'segmentation', 'de-segmentation']
complex_options = ['color_assisted', 'masking', 'de-masking']

mapping_func = preprocessing.load_images


# todo: solve multiple input paths
def get_dataset(path, option, *args):

    option = option.lower()
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if option in simple_options:
        if option == 'reconstruction':
            x_path = path + path_temples_ruins
            y_path = path + path_temples
        elif option == 'color_reconstruction':
            x_path = path + path_temples_ruins_colors
            y_path = path + path_temples_colors
        elif option == 'segmentation':
            x_path = [path + path_temples_colors, path + path_temples_ruins_colors]
            y_path = [path + path_temples, path + path_temples_ruins]
        elif option == 'de-segmentation':
            x_path = [path + path_temples, path + path_temples_ruins]
            y_path = [path + path_temples_colors, path + path_temples_ruins_colors]
        else:
            x_path = y_path = None

        return reconstruction(*args, x_path, y_path)

    elif option in complex_options:
        if option == 'color_assisted':
            pass
        # for masking and de-masking
        # x is the segmented temple ruins
        # y is the segmented, complete temple
        # z is the true-color temple
        elif option == 'masking':
            pass

        elif option == 'de-masking':
            preprocessing.apply_mask = True
            x_path = path + path_temples_ruins_colors
            y_path = path + path_temples_colors
            z_path = path + path_temples
            return reconstruction(*args, x_path, y_path, z_path)

    else:
        raise Exception('Option not supported. Run train.py -h to see the supported options.')

    return reconstruction(*args, *paths)


def reconstruction(temples, split=0.25, batch_size=1, buffer_size=400, *paths):
    files = list(map(lambda x: simple(x, paths), temples))
    files = reduce(concat, files)
    train_files, val_files = train_val_split(files, split, buffer_size)

    train = train_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)
    val = val_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)

    return train, val


def simple(number, paths):
    pattern = f'/*temple{number}*/*'
    file_datasets = [tf.data.Dataset.list_files(path + pattern, shuffle=False).repeat(rep)
                     for path, rep in zip(paths, repetitions)]
    return tf.data.Dataset.zip(tuple(file_datasets))


def concat(a, b):
    return a.concatenate(b)


def train_val_split(dataset, split, buffer_size=400):
    dataset = dataset.shuffle(buffer_size)
    train = dataset.skip(round(len_dataset * split))
    val = dataset.take(round(len_dataset * split))
    return train, val
