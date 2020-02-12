import tensorflow as tf
import datetime
import glob

from utils import pix2pix_preprocessing as preprocessing
import numpy as np


class Reconstructor:
    def __init__(self, segmenter='path', desegmenter='path', reconstructor='path', log_dir='logs'):
        self.segmenter = tf.keras.models.load_model(segmenter)
        self.reconstructor = tf.keras.models.load_model(reconstructor)
        self.desegmenter = tf.keras.models.load_model(desegmenter)

        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        predict_log_dir = log_dir + r'\\' + current_time + r'\predict'
        self.predict_writer = tf.summary.create_file_writer(predict_log_dir)

        self._image_dir = 'images'
        self._image_list = []

        self.step = 0

    def set_image_dir(self, image_dir: str):
        self._image_dir = image_dir
        self._image_list = glob.glob(self._image_dir + r'\*')

    def get_image_batch(self, start, finish):
        batch = tf.data.Dataset.list_files(self._image_list[start:finish])
        batch = batch.map(preprocessing.load_images_predict).batch(1)

        return batch

    def get_random_batch(self, ammount):
        choice = np.random.choice(len(self._image_list), ammount)
        batch = [self._image_list[i] for i in choice]
        batch = tf.data.Dataset.list_files(batch)
        batch = batch.map(preprocessing.load_images_predict).batch(1)

        return batch

    def predict(self, dataset):
        for x in dataset:
            # x is the input
            segmented = self.segmenter(x, training=False)
            reconstructed = self.reconstructor(segmented, training=False)
            desegmented = self.desegmenter(reconstructed, training=False)
            stack = tf.stack([x, segmented, reconstructed, desegmented], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with self.predict_writer.as_default():
                tf.summary.image('predictions', stack, step=self.step, max_outputs=4)

            self.step += 1
