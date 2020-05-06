import os
import tensorflow as tf
from functools import reduce
from utils import preprocessing

path_dataset = 'dataset'
path_temples = '/temples'
path_temples_ruins = '/temples_ruins'
path_temples_colors = '/colors_temples'
path_temples_ruins_colors = '/colors_temples_ruins'

x_path = y_path = aux_path = None

repeat = 1
len_dataset = 100


def setup(path, option, *args):
    global x_path, y_path

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if option.lower() == 'reconstruction':
        x_path = path + path_temples_ruins
        y_path = path + path_temples
        pass
    elif option.lower() == 'color_reconstruction':
        x_path = path + path_temples_ruins_colors
        y_path = path + path_temples_colors
        pass
    elif option.lower() == 'segmentation':
        x_path = [path + path_temples_colors, path + path_temples_ruins_colors]
        y_path = [path + path_temples, path + path_temples_ruins]
        pass
    elif option.lower() == 'segmentation_inv':
        x_path = [path + path_temples, path + path_temples_ruins]
        y_path = [path + path_temples_colors, path + path_temples_ruins_colors]
        pass
    elif option.lower() == 'color_assisted':
        pass


def reconstruction(temples, split=0.25, batch_size=1, buffer_size=400):
    files = list(map(simple_xy, temples))
    files = reduce(concat, files)
    train_files, val_files = train_val_split(files, split, buffer_size)

    train = train_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)
    val = val_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)

    return train, val


def assisted(temples, split=0.25, batch_size=1, buffer_size=400):
    files = list(map(simple_xy, temples))


def simple_xy(number):
    pattern = f'/*temple_{number}*/*'
    x_paths, y_paths = list(x_path), list(y_path)
    x_patterns = list(map(lambda z: z + pattern, x_paths))
    y_patterns = list(map(lambda z: z + pattern, y_paths))

    x = tf.data.Dataset.list_files(x_patterns, shuffle=False)
    y = tf.data.Dataset.list_files(y_patterns, shuffle=False)
    y = y.repeat(repeat)
    combined = tf.data.Dataset.zip((x, y))

    return combined


def concat(a, b):
    return a.concatenate(b)


def train_val_split(dataset, split, buffer_size=400):
    dataset = dataset.shuffle(buffer_size)
    train = dataset.skip(round(len_dataset * split))
    val = dataset.take(round(len_dataset * split))
    return train, val
