from utils import custom_preprocessing as cp
from utils import preprocessing as pp
import tensorflow as tf
import glob

path_temples = '/temples'
path_temples_ruins = '/temples_ruins'
path_temples_colors = '/colors_temples'
path_temples_ruins_colors = '/colors_temples_ruins'
images_per_temple = 300


def get_dataset(option, args):
    if option.lower() == 'reconstruction':
        return get_dataset_reconstruction(*args)
    elif option.lower() == 'color_reconstruction':
        return get_dataset_reconstruction(*args, mode='color')
    elif option.lower() == 'segmentation':
        return get_dataset_segmentation(*args)
    elif option.lower() == 'segmentation_inv':
        return get_dataset_segmentation(*args, inverse=True)
    elif option.lower() == 'color_assisted':
        return get_dataset_dual_input(*args)


def setup_paths(path_dataset):
    global path_temples
    global path_temples_ruins
    global path_temples_colors
    global path_temples_ruins_colors

    path_temples = path_dataset + path_temples
    path_temples_ruins = path_dataset + path_temples_ruins
    path_temples_colors = path_dataset + path_temples_colors
    path_temples_ruins_colors = path_dataset + path_temples_ruins_colors


def simple_dataset(input_path: str, output_path: str, split=0.25, batch_size=1, img_format='png'):
    input_files = glob.glob(input_path + f'/*.{img_format}')
    output_files = glob.glob(output_path + f'/*.{img_format}')
    shuffle_size = len(input_files)

    validation_size = round(shuffle_size * split)

    dataset_input = tf.data.Dataset.from_tensor_slices(input_files)
    dataset_output = tf.data.Dataset.from_tensor_slices(output_files)
    dataset = tf.data.Dataset.zip((dataset_input, dataset_output)).shuffle(shuffle_size)

    train, validation = _split_dataset(dataset, validation_size, batch_size)

    return train, validation


def get_dataset_prediction(input_path: str, img_format='png'):
    input_files = tf.data.Dataset.list_files(input_path + f'/*.{img_format}')
    dataset = input_files.map(cp.load_images_val).batch(1)
    return dataset


def get_dataset_dual_input(temples: list, split=0.25, batch_size=1, repeat=1, img_format='png'):
    buffer_size = len(temples) * images_per_temple * repeat
    validation_size = round(buffer_size * split)

    dataset_paths = []
    for i, temple in enumerate(temples):
        glob_pattern = f'/*{temple}*/*.{img_format}'
        dataset_ruins = tf.data.Dataset.list_files(path_temples_ruins + glob_pattern, shuffle=False)
        dataset_colors = tf.data.Dataset.list_files(path_temples_colors + glob_pattern, shuffle=False)
        dataset_temples = tf.data.Dataset.list_files(path_temples + glob_pattern, shuffle=False)

        dataset_colors = dataset_colors.repeat(repeat)
        dataset_temples = dataset_temples.repeat(repeat)

        combined = tf.data.Dataset.zip((dataset_ruins, dataset_colors, dataset_temples))
        dataset_paths.append(combined)

    train, validation = _concat_datasets(dataset_paths, validation_size, buffer_size, batch_size)

    return train, validation


def get_dataset_reconstruction(temples: list, split=0.25, batch_size=1, repeat=1, img_format='png', mode='real'):
    """Aimed at obtaining a reconstruction dataset

    :param temples: list. Temple list.
    :param split: float. Validation split.
    :param batch_size: int. Batch size.
    :param repeat: int. Ratio of ruins per temple.
    :param img_format: string. Image format.
    :param mode: string. Either real or color. Wheter or not to obtain a segmented version of the ruins.
    :return: train, validation. Tensorflow datasets.
    """
    buffer_size = len(temples) * images_per_temple * repeat
    validation_size = round(buffer_size * split)

    dataset_paths = []
    for n in temples:
        glob_pattern = f'/*temple_{n}*/*.{img_format}'

        if mode == 'real':
            dataset_in = tf.data.Dataset.list_files(path_temples_ruins + glob_pattern, shuffle=False)
            dataset_out = tf.data.Dataset.list_files(path_temples + glob_pattern, shuffle=False)

        elif mode == 'color':
            dataset_in = tf.data.Dataset.list_files(path_temples_ruins_colors + glob_pattern, shuffle=False)
            dataset_out = tf.data.Dataset.list_files(path_temples_colors + glob_pattern, shuffle=False)

        else:
            raise Exception('Unsupported method. Supported methods are real and color')

        dataset_in = dataset_in.repeat(repeat)

        combined = tf.data.Dataset.zip((dataset_in, dataset_out))
        dataset_paths.append(combined)

    train, validation = _concat_datasets(dataset_paths, validation_size, buffer_size, batch_size)

    return train, validation


def get_dataset_segmentation(temples: list, split=0.25, batch_size=1, repeat=1, img_format='png', inverse=False,
                             mask=False):
    buffer_size = len(temples) * images_per_temple * repeat
    validation_size = round(buffer_size * split)

    file_patterns_colors = []
    file_patterns_real = []
    for temple in temples:
        glob_pattern = f'/*{temple}*/*.{img_format}'
        file_patterns_colors.append(path_temples_colors + glob_pattern)
        file_patterns_colors.append(path_temples_ruins_colors + glob_pattern)
        file_patterns_real.append(path_temples + glob_pattern)
        file_patterns_real.append(path_temples_ruins + glob_pattern)

    dataset_colors = tf.data.Dataset.list_files(file_patterns_colors, shuffle=False)
    dataset_real = tf.data.Dataset.list_files(file_patterns_real, shuffle=False)

    if not inverse:
        dataset = tf.data.Dataset.zip((dataset_real, dataset_colors)).shuffle(buffer_size)
    else:
        dataset = tf.data.Dataset.zip((dataset_colors, dataset_real)).shuffle(buffer_size)

    if mask:
        train, validation = _mask_outputs(dataset, validation_size, batch_size)
    else:
        train, validation = _split_dataset(dataset, validation_size, batch_size)

    return train, validation


def get_dataset_mask(dir_in, dir_out, batch_size=1):
    x = tf.data.Dataset.list_files(dir_in + '/*.png', shuffle=False)
    y = tf.data.Dataset.list_files(dir_out + '/*', shuffle=False)
    dataset = tf.data.Dataset.zip((x, y))
    dataset = dataset.map(pp.load_images).batch(batch_size)
    return dataset


def _concat_datasets(dataset_paths, validation_size, buffer_size, batch_size):
    dataset = dataset_paths[0]
    dataset_paths.pop(0)
    for single_dataset in dataset_paths:
        dataset = dataset.concatenate(single_dataset)
    dataset = dataset.shuffle(buffer_size)

    return _split_dataset(dataset, validation_size, batch_size)


def _split_dataset(dataset, validation_size, batch_size):
    validation = dataset.take(validation_size).map(cp.load_images_val).batch(batch_size)
    train = dataset.skip(validation_size).map(pp.load_images).batch(batch_size)

    return train, validation


def _mask_outputs(dataset, validation_size, batch_size):
    validation = dataset.take(validation_size).map(cp.load_images_mask_val).batch(batch_size)
    train = dataset.skip(validation_size).map(cp.load_images_mask_train).batch(batch_size)

    return train, validation


if __name__ == '__main__':
    path_in = '../dataset/colors_temples_ruins/colors_temple_0_ruins_1'
    path_out = '../dataset/colors_temples/colors_temple_0'
    ds = get_dataset_mask(path_in, path_out)
