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

simple_options = ['reconstruction', 'color_reconstruction', 'segmentation', 'de-segmentation']
complex_options = ['color_assisted', 'masking', 'de-masking']

mapping_func = preprocessing.load_images


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
        # todo: solve multiple input paths
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
            x_path = path + path_temples_ruins
            y_path = path + path_temples_colors
            z_path = path + path_temples

        elif option in ['masking', 'de-masking']:
            preprocessing.apply_mask = True

            x_path = path + path_temples_ruins_colors
            y_path = path + path_temples_colors
            z_path = path + path_temples

            if option == 'de-masking':
                preprocessing.demasking = True

            if option == 'masking':
                aux_path = path + path_temples_ruins
                return reconstruction(*args, x_path, y_path, z_path, aux_path)

        else:
            x_path = y_path = z_path = None

        return reconstruction(*args, x_path, y_path, z_path)

    else:
        raise Exception('Option not supported. Run train.py -h to see the supported options.')


def reconstruction(temples, split=0.25, batch_size=1, buffer_size=400, *paths):
    files = list(map(lambda x: simple(x, paths), temples))
    files = reduce(concat, files)

    # dataset size
    size = list(map(lambda x: len(glob.glob(paths[0] + f'/*temple_{x}*/*')), temples))
    size = reduce((lambda x, y: x + y), size)

    train_files, val_files = train_val_split(files, split, size)

    train = train_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)
    val = val_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)

    return train, val


def simple(number, paths):
    pattern = f'/*temple_{number}*/*'
    file_datasets = [tf.data.Dataset.list_files(path + pattern, shuffle=False).repeat(rep)
                     for path, rep in zip(paths, repetitions)]
    return tf.data.Dataset.zip(tuple(file_datasets))


def concat(a, b):
    return a.concatenate(b)


def train_val_split(dataset, split, size):
    dataset = dataset.shuffle(size)
    train = dataset.skip(round(size * split))
    val = dataset.take(round(size * split))
    return train, val


if __name__ == '__main__':
    ds_args = [[0], 0.25, 1, 300]
    ds_path = '../dataset'
    preprocessing.apply_mask = True
    repetitions = [1, 2, 2]
    get_dataset(ds_path, 'masking', *ds_args)

