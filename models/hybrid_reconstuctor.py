import datetime
import glob

import numpy as np
import tensorflow as tf
from models.pix2pix import Pix2Pix
from utils import custom_preprocessing as cp
from PIL import Image


class HybridReconstuctor(Pix2Pix):
    def get_dataset(self, *, temples, ruins_per_temple=1, dataset_path=None, split=0.25, shuffle=True):
        if dataset_path is None:
            dataset_path = r'..\dataset\\'

        buffer_size = len(temples) * ruins_per_temple * 300

        datasets = []
        for i, temple in enumerate(temples):
            ruins_path = dataset_path + r'\temples_ruins\\' + temple + '*'
            colors_path = dataset_path + r'\colors_temples\colors_' + temple
            temple_path = dataset_path + r'\temples\\' + temple

            datasets.append(self.get_single_dataset(ruins_path, temple_path, colors_path))

        train_dataset = datasets[0]
        datasets.pop(0)
        for dataset in datasets:
            train_dataset = train_dataset.concatenate(dataset)

        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size)

        # train/val split
        train_size = buffer_size - round(buffer_size * split)
        train = train_dataset.take(train_size).map(cp.load_images_train).batch(1)
        validation = train_dataset.skip(train_size).map(cp.load_images_test).batch(1)

        return train, validation

    @staticmethod
    def get_single_dataset(ruins_path, temple_path, colors_path):
        ruins_path_list = glob.glob(ruins_path + r'\*.png')
        colors_path_list = glob.glob(colors_path + r'\*.png')
        temple_path_list = glob.glob(temple_path + r'\*.png')

        repetition = len(ruins_path_list) // len(temple_path_list)

        ruins_dataset = tf.data.Dataset.list_files(ruins_path_list, shuffle=False)
        temple_dataset = tf.data.Dataset.list_files(temple_path_list, shuffle=False)
        colors_dataset = tf.data.Dataset.list_files(colors_path_list, shuffle=False)

        if repetition > 1:
            temple_dataset = temple_dataset.repeat(repetition)
            colors_dataset = colors_dataset.repeat(repetition)

        dataset = tf.data.Dataset.zip((ruins_dataset, temple_dataset, colors_dataset))

        return dataset

    @staticmethod
    def get_prediction_dataset(*paths):
        # Survey
        path_collection = []
        for path in paths:
            path_collection.append(glob.glob(path + '\\*.png'))

        datasets = []
        for file_dataset in path_collection:
            datasets.append(tf.data.Dataset.list_files(file_dataset, shuffle=False))

        dataset = tf.data.Dataset.zip(tuple(datasets))
        dataset = dataset.map(cp.load_images_test).batch(1)

        return dataset

    @tf.function
    def train_step(self, ruin, temple, color):
        with tf.GradientTape(persistent=True) as tape:
            gen_output = self.generator([ruin, color], training=True)

            disc_real = self.discriminator([ruin, temple], training=True)
            disc_generated = self.discriminator([ruin, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated, gen_output, temple)
            disc_loss = self.discriminator_loss(disc_real, disc_generated)

        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        if self.log_dir is not None:
            self.train_disc_loss(disc_loss)
            self.train_gen_loss(gen_loss)
            self.train_real_acc(tf.ones_like(disc_real), disc_real)
            self.train_gen_acc(tf.zeros_like(disc_generated), disc_generated)

    def fit(self, train_ds, test_ds=None, epochs=100):
        for epoch in range(epochs):
            # Train
            for ruin, temple, color in train_ds:
                self.train_step(ruin, temple, color)

            if test_ds is not None:
                self.validate(test_ds)

            self._metric_update(train_ds, test_ds, epoch)
            if self.log_dir is not None:
                self._train_predict(train_ds, self.train_summary_writer, epoch, 'train')
                if test_ds is not None:
                    self._train_predict(test_ds, self.val_summary_writer, epoch, 'validation')

    # prediction methods
    def predict(self, dataset, log_path, samples):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        writer = self._get_summary_writer(log_path, current_time, 'predict')

        step = 0
        if samples == 'all':
            target = dataset
        else:
            target = dataset.take(samples)

        for x, y, z in target:
            # x and z are the inputs
            prediction = self.generator([x, z], training=False)
            stack = tf.stack([x, z, prediction, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('predictions', stack, step=step, max_outputs=4)

            step += 1

    def validate(self, test):
        for test_ruin, test_temple, test_color in test:
            gen_output = self.generator([test_ruin, test_color], training=False)

            disc_real_output = self.discriminator([test_ruin, test_temple], training=False)
            disc_generated_output = self.discriminator([test_ruin, gen_output], training=False)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, test_temple)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            if self.log_dir is not None:
                self.val_disc_loss(disc_loss)
                self.val_gen_loss(gen_loss)
                self.val_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)
                self.val_real_acc(tf.ones_like(disc_real_output), disc_real_output)

    def _train_predict(self, dataset, writer, step, name='train'):
        for ruin, temple, color in dataset.take(1):
            generated = self.generator([ruin, color], training=False)
            stack = tf.stack([ruin, color, generated, temple], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image(name, stack, step=step, max_outputs=4)


def main(training_name, temples):
    # TODO: standarise r and f
    log_path = f'..\\logs\\{training_name}'
    ds_path = r'..\dataset'
    cp.RESIZE_FACTOR = 1.3

    reconstructor = HybridReconstuctor(log_dir=log_path, autobuild=False)

    reconstructor.build_generator(heads=2, inplace=True)
    reconstructor.build_discriminator(inplace=True)

    train, validation = reconstructor.get_dataset(temples=temples, dataset_path=ds_path, split=0.25)
    reconstructor.fit(train, validation, epochs=50)
    tf.keras.models.save_model(reconstructor.generator, f'../trained_models/{training_name}.h5')


# TODO: simplify/unify prediction functions and methods
def predict_batch(target='temple_0', ruins=1):
    temple = target

    log_path = r'..\logs\survey\pairs'

    ds_path = r'..\dataset\\'
    ruins = ds_path + r'temples_ruins\\' + temple + f'_ruins_{ruins}'
    colors = ds_path + r'colors_temples\colors_' + temple
    temples = ds_path + r'temples\\' + temple

    reconstructor = HybridReconstuctor(gen_path='../trained_models/reconstructor.h5', autobuild=False)
    predict_ds = reconstructor.get_prediction_dataset(ruins, temples, colors)

    reconstructor.predict(predict_ds, log_path, samples='all')


if __name__ == '__main__':
    predict_batch(target='temple_0', ruins=1)
