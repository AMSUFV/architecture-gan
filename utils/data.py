import glob
import os
import tensorflow as tf

from datetime import datetime
from functools import reduce
from utils import preprocessing, text

seed = datetime.now().microsecond

PATH_TEXTS = '/textos_parrafos'

PATH_TEMPLES = '/temples'
PATH_TEMPLES_RUINS = '/temples_ruins'
PATH_TEMPLES_COLORS = '/colors_temples'
PATH_TEMPLES_RUINS_COLORS = '/colors_temples_ruins'

repetitions = [1, 2]

mapping_func = preprocessing.load_images
glob_pattern = '/*temple_{}*/*'


def get_dataset(path, option, *args):

    option = option.lower()
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if option == 'reconstruction':
        x_path = path + PATH_TEMPLES_RUINS
        y_path = path + PATH_TEMPLES
        return reconstruction(*args, x_path, y_path)

    elif option == 'color_reconstruction':
        x_path = path + PATH_TEMPLES_RUINS_COLORS
        y_path = path + PATH_TEMPLES_COLORS
        return reconstruction(*args, x_path, y_path)

    elif option == 'color_assisted':
        x_path = path + PATH_TEMPLES_RUINS
        y_path = path + PATH_TEMPLES
        z_path = path + PATH_TEMPLES_COLORS
        return reconstruction(*args, x_path, y_path, z_path)

    elif option in ['masking', 'de-masking']:
        preprocessing.apply_mask = True
        x_path = path + PATH_TEMPLES_RUINS_COLORS
        y_path = path + PATH_TEMPLES_COLORS
        z_path = path + PATH_TEMPLES
        if option == 'de-masking':
            preprocessing.demasking = True
            return reconstruction(*args, x_path, y_path, z_path)
        if option == 'masking':
            aux_path = path + PATH_TEMPLES_RUINS
            return reconstruction(*args, x_path, y_path, z_path, aux_path)

    elif option == 'segmentation':
        x_path = path + PATH_TEMPLES
        y_path = path + PATH_TEMPLES_COLORS
        return reconstruction(*args, x_path, y_path)

    elif option == 'de-segmentation':
        x_path = path + PATH_TEMPLES_COLORS
        y_path = path + PATH_TEMPLES
        return reconstruction(*args, x_path, y_path)

    elif option == 'text_assisted':
        x_path = path + PATH_TEMPLES_RUINS
        y_path = path + PATH_TEMPLES
        text_path = path + PATH_TEXTS
        return reconstruction(*args, x_path, y_path, descriptions=True, text_path=text_path)

    else:
        raise Exception('Option not supported. Run keras_train.py -h to see the supported options.')


def reconstruction(temples, split=0.25, batch_size=1, buffer_size=400, *paths, **kwargs):

    # dataset size
    size = list(map(lambda x: len(glob.glob(paths[0] + glob_pattern.format(x))), temples))
    size = reduce((lambda x, y: max(x, y)), size)

    files = list(map(lambda x: get_unique(x, paths), temples))
    files = reduce(concat, files)

    train_files, val_files = train_val_split(files, split, size, buffer_size)

    train = train_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)
    val = val_files.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)

    # embeddings
    if kwargs['descriptions']:
        embeddings = get_embeddings(temples, kwargs['text_path'], repeat=size//len(temples))
        train_emb, val_emb = train_val_split(embeddings, split, size, buffer_size)
        train_emb = train_emb.batch(batch_size)
        val_emb = val_emb.batch(batch_size)
        train = tf.data.Dataset.zip((train, train_emb))
        val = tf.data.Dataset.zip((val, val_emb))

    return train, val


def get_unique(number, paths):
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
    dataset = dataset.shuffle(buffer_size, seed=seed)
    train = dataset.skip(round(size * split))
    val = dataset.take(round(size * split))
    return train, val


def get_embeddings(temples, path, repeat):

    dataset = []
    for temple in temples:
        file_path = path + f'/caso{temple + 1}.sent.txt'

        embeddings = []
        with open(file_path, 'r') as file:
            for line in file:
                embeddings.append(text.tokenize(line))
        embeddings = tf.stack(embeddings)
        embeddings = tf.reduce_mean(embeddings, axis=0)
        embeddings = tf.repeat(embeddings, repeat, axis=0)
        dataset.append(tf.data.Dataset.from_tensor_slices(embeddings))

    return reduce(concat, dataset)


def get_simple_dataset(width, height, *paths):
    preprocessing.width = width
    preprocessing.height = height

    file_dataset = [tf.data.Dataset.list_files(path, shuffle=False)
                    for path in paths]
    file_dataset = tf.data.Dataset.zip(tuple(file_dataset))

    return file_dataset.map(preprocessing.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)


def validate(model, width, height, down_blocks):
    if model.lower() == 'pix2pix':
        if width % 2**down_blocks != 0:
            raise Exception("Width exception. The image won't make it through the bottleneck.")
        elif height % 2**down_blocks != 0:
            raise Exception("Height exception. The image won't make it through the bottleneck.")
        else:
            pass
