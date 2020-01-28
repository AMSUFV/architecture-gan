import glob
import tensorflow as tf
from models.pix2pix import Pix2Pix
from models.utils import custom_preprocessing as cp


class HybridReconstuctor(Pix2Pix):
    def get_dataset(self, temples, split=0.2, dataset_path=None):
        if dataset_path is None:
            dataset_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\\'

        for i, temple in enumerate(temples):

            ruins_path = dataset_path + r'temples_ruins\\' + temple + '*'
            colors_path = dataset_path + r'colors_temples\colors_' + temple
            temple_path = dataset_path + r'temples\\' + temple

            if i == 0:
                train_dataset = self.get_single_dataset(ruins_path, colors_path, temple_path)
            else:
                tr = self.get_single_dataset(ruins_path, colors_path, temple_path)
                train_dataset = train_dataset.concatenate(tr)

        return train_dataset

    @staticmethod
    def get_single_dataset(ruins_path, colors_path, temple_path, split=0.2):
        ruins_path_list = glob.glob(ruins_path + r'\*.png')
        colors_path_list = glob.glob(colors_path + r'\*.png')
        temple_path_list = glob.glob(temple_path + r'\*.png')

        batch_size = 1
        buffer_size = len(ruins_path_list)

        repetition = len(temple_path_list) // len(ruins_path_list)

        ruins_dataset = tf.data.Dataset.list_files(ruins_path_list, shuffle=False)
        colors_dataset = tf.data.Dataset.list_files(colors_path_list, shuffle=False)
        temple_dataset = tf.data.Dataset.list_files(temple_path_list, shuffle=False)

        colors_dataset = colors_dataset.repeat(repetition)
        temple_dataset = temple_dataset.repeat(repetition)

        train_dataset = tf.data.Dataset.zip((ruins_dataset, colors_dataset, temple_dataset))
        train_dataset = train_dataset.map(cp.load_images_train).shuffle(buffer_size).batch(batch_size)

        return train_dataset


if __name__ == '__main__':
    reconstructor = HybridReconstuctor(log_dir=r'logs\\test')
    train = reconstructor.get_dataset(['temple_0'])