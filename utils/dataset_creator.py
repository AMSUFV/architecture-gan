from utils import custom_preprocessing as cp
import tensorflow as tf
import glob

path_temples = '../dataset/temples'
path_temples_ruins = '../dataset/temples_ruins'
path_temples_colors = '../dataset/colors_temples'
path_temples_ruins_colors = '../dataset/colors_temples_ruins'
images_per_temple = 300


def simple_dataset(input_path: str, output_path: str, split=0.25, batch_size=1, img_format='png'):
    input_files = glob.glob(input_path + f'/*.{img_format}')
    output_files = glob.glob(output_path + f'/*.{img_format}')
    shuffle_size = len(input_files)

    validation_size = round(shuffle_size * split)

    dataset_input = tf.data.Dataset.from_tensor_slices(input_files)
    dataset_output = tf.data.Dataset.from_tensor_slices(output_files)
    dataset = tf.data.Dataset.zip((dataset_input, dataset_output)).shuffle(shuffle_size)

    validation = dataset.take(validation_size).map(cp.load_images_test).batch(batch_size)
    train = dataset.skip(validation_size).map(cp.load_images_train).batch(batch_size)

    return train, validation


def custom_dataset(temples: list, split=0.25, batch_size=1, repeat=1, img_format='png'):
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

    dataset = dataset_paths[0]
    dataset_paths.pop(0)
    for single_dataset in dataset_paths:
        dataset = dataset.concatenate(single_dataset)

    dataset = dataset.shuffle(buffer_size)
    validation = dataset.take(validation_size).map(cp.load_images_test).batch(batch_size)
    train = dataset.skip(validation_size).map(cp.load_images_train).batch(batch_size)

    return train, validation
