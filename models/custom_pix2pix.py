import glob
import tensorflow as tf

from random import shuffle

from models.pix2pix import Pix2Pix
from utils import custom_preprocessing as cp


class CustomPix2Pix(Pix2Pix):
    # def get_complete_datset(self, temples, ruins_per_temple=1, mode=None):
    #     """
    #     Este método asume una estructura de archivos en la que los templos completos están en una carpeta llamada
    #     temples y llamados temple_0, temple_1, etc, y sus ruinas en la carpeta temples_ruins
    #     :param mode:
    #     :param temples:
    #     :param ruins_per_temple:
    #     :return:
    #     """
    #
    #     if mode is None:
    #         mode = 'picture_reconstruction'
    #
    #     # ruins_to_temples
    #     dataset_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\\'
    #     for i, temple in enumerate(temples):
    #         if mode == 'to_colors':
    #             input_path = dataset_path + r'temples*\\' + temple + '*'
    #             output_path = dataset_path + r'colors*\colors_' + temple + '*'
    #
    #         elif mode == 'to_pictures':
    #             # color temples and ruins to real
    #             input_path = dataset_path + r'colors*\colors_' + temple + '*'
    #             output_path = dataset_path + r'temples*\\' + temple + '*'
    #
    #         elif mode == 'picture_reconstruction':
    #             input_path = dataset_path + r'temples_ruins\\' + temple + '*'
    #             output_path = dataset_path + r'temples\\' + temple
    #
    #         elif mode == 'color_reconstruction':
    #             input_path = dataset_path + r'colors_temples_ruins\\colors_' + temple + '*'
    #             output_path = dataset_path + r'colors_temples\\colors_' + temple
    #
    #         if i == 0:
    #             train_dataset, val_dataset = self.get_dataset(input_path, output_path, ruins_per_temple)
    #         else:
    #             tr, val = self.get_dataset(input_path, output_path, ruins_per_temple)
    #             train_dataset = train_dataset.concatenate(tr)
    #             val_dataset = val_dataset.concatenate(val)
    #
    #     return train_dataset, val_dataset
    #
    # # dataset creation function
    # @staticmethod
    # def get_dataset(input_path, output_path):
    #     """Generación del dataset. Orientado a la extracción de diferentes ángulos de templos griegos
    #
    #     :param input_path: ruta a las imágenes de ruinas de templos
    #     :param output_path: ruta a las imágenes de templos completos
    #     :param repeat_real: el número de veces que las imágenes de templos completos se repiten; tantas como diferentes
    #                         modelos de sus ruinas se tengan
    #     :return: train_dataset, test_datset
    #     """
    #     batch_size = 1
    #
    #     input_path = glob.glob(input_path + r'\*.png')
    #     output_path = glob.glob(output_path + r'\*.png')
    #
    #     repeat_real = len(output_path) // len(input_path)
    #
    #     buffer_size = min(len(input_path), len(output_path))
    #
    #     train_mask = ([True] * (len(output_path) // 100 * 8) + [False] * (len(output_path) // 100 * 2)) * 10
    #     test_mask = ([False] * (len(output_path) // 100 * 8) + [True] * (len(output_path) // 100 * 2)) * 10
    #
    #     train_input = list(compress(input_path, train_mask * repeat_real))
    #     train_real = list(compress(output_path, train_mask))
    #
    #     test_input = list(compress(input_path, test_mask * repeat_real))
    #     test_real = list(compress(output_path, test_mask))
    #
    #     # train
    #     input_dataset = tf.data.Dataset.list_files(train_input, shuffle=False)
    #     output_dataset = tf.data.Dataset.list_files(train_real, shuffle=False)
    #     output_dataset = output_dataset.repeat(repeat_real)
    #
    #     train_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    #     train_dataset = train_dataset.map(preprocessing.load_images_train).shuffle(buffer_size).batch(batch_size)
    #
    #     # test
    #     test_input_ds = tf.data.Dataset.list_files(test_input, shuffle=False)
    #     test_real_ds = tf.data.Dataset.list_files(test_real, shuffle=False)
    #     test_real_ds = test_real_ds.repeat(repeat_real)
    #
    #     test_dataset = tf.data.Dataset.zip((test_input_ds, test_real_ds))
    #     test_dataset = test_dataset.map(preprocessing.load_images_test).batch(batch_size)
    #
    #     return train_dataset, test_dataset

    def get_dataset(self, temples, dataset_path=None, split=0.2, ruins_per_temple=2, image_shape=None):
        if image_shape is not None:
            cp.IMG_WIDTH, cp.IMG_HEIGHT = image_shape

        if dataset_path is None:
            dataset_path = r'..\dataset\\'

        buffer_size = len(temples) * ruins_per_temple * 300

        datasets = []
        for i, temple in enumerate(temples):
            ruins_path = dataset_path + r'\temples_ruins\\' + temple + '*'
            temple_path = dataset_path + r'\temples\\' + temple

            datasets.append(self.get_single_dataset(ruins_path, temple_path))

        train_dataset = datasets[0]
        datasets.pop(0)
        for dataset in datasets:
            train_dataset = train_dataset.concatenate(dataset)
        train_dataset = train_dataset.shuffle(buffer_size)

        # train/val split
        train_size = buffer_size - round(buffer_size * split)
        val_size = buffer_size - train_size

        train = train_dataset.take(train_size).map(cp.load_images_train)
        train = train.shuffle(train_size, reshuffle_each_iteration=False).batch(1)

        validation = train_dataset.skip(train_size).map(cp.load_images_test)
        validation = validation.shuffle(val_size, reshuffle_each_iteration=False).batch(1)

        return train, validation

    @staticmethod
    def get_single_dataset(ruins_path, temple_path):
        ruins_path_list = glob.glob(ruins_path + r'\*.png')
        temple_path_list = glob.glob(temple_path + r'\*.png')

        repetition = len(ruins_path_list) // len(temple_path_list)

        ruins_dataset = tf.data.Dataset.list_files(ruins_path_list, shuffle=False)
        temple_dataset = tf.data.Dataset.list_files(temple_path_list, shuffle=False)

        temple_dataset = temple_dataset.repeat(repetition)

        full_dataset = tf.data.Dataset.zip((ruins_dataset, temple_dataset))

        return full_dataset


if __name__ == '__main__':

    log_path = r'..\logs\full_temple_train'
    ds_path = r'..\dataset'
    temple_list = ['temple_1', 'temple_2', 'temple_3', 'temple_4', 'temple_5', 'temple_6',
                   'temple_7', 'temple_8', 'temple_9']
    cp.RESIZE_FACTOR = 1.3

    pix2pix = CustomPix2Pix(log_dir=r'..\logs\full_temple_train_pix2pix', autobuild=True)
    train_ds, validation_ds = pix2pix.get_dataset(temples=temple_list, dataset_path=ds_path, split=0.25)
    pix2pix.fit(train_ds, epochs=50)
    tf.keras.models.save_model(pix2pix.generator, '../trained_models/reconstructor_simple.h5')
