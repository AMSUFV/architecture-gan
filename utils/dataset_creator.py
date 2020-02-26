from utils import custom_preprocessing as cp
import tensorflow as tf
import glob


def simple_dataset(input_path: str, output_path: str, split=0.25, img_format='png', batch_size=1):
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


def custom_dataset(temples: list, split=0.25, repeat=1, dataset_path=None):
    pass
