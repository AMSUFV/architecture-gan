from __future__ import absolute_import, division, print_function, unicode_literals
# TODO: Make sure everything goes where it should go; logs, saved models, etc.
import datetime
import numpy as np
import tensorflow as tf
# creación del dataset
import glob
from itertools import compress
from sklearn.model_selection import train_test_split

# modelo base y preprocesamiento
from utils import pix2pix_preprocessing as preprocessing
from utils import custom_preprocessing as cp
from utils.basemodel import BaseModel


# Convenience functions
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# TODO add training checkpoints
class Pix2Pix:
    def __init__(self, *, gen_path=None, disc_path=None, log_dir=None):

        self.generator = self._set_weights(gen_path, self.build_generator, 'initial_generator')
        self.discriminator = self._set_weights(disc_path, self.build_discriminator, 'initial_discriminator')

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.LAMBDA = 100

        # TODO: Single responsibility; sacar las métricas
        # Tensorboard
        self.log_dir = log_dir
        if self.log_dir is not None:
            self.train_summary_writer = self._set_logdir(self.log_dir, 'train')
            self.val_summary_writer = self._set_logdir(self.log_dir, 'validation')

            # Métricas
            self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            self.train_real_acc = tf.keras.metrics.BinaryAccuracy('train_real_accuracy')
            self.train_gen_acc = tf.keras.metrics.BinaryAccuracy('train_gen_accuracy')

            self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
            self.val_gen_acc = tf.keras.metrics.BinaryAccuracy('val_gen_accuracy')
            self.val_real_acc = tf.keras.metrics.BinaryAccuracy('val_real_accuracy')

            # generator loss
            self.train_gen_loss = tf.keras.metrics.Mean('train_gen_loss', dtype=tf.float32)
            self.val_gen_loss = tf.keras.metrics.Mean('val_gen_loss', dtype=tf.float32)

    @staticmethod
    def _set_logdir(path, name='train'):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = rf'{path}\\{current_time}\\{name}'
        writer = tf.summary.create_file_writer(log_path)
        return writer

    def _set_weights(self, path, func, name='model', save_weights=False):
        if path is not None:
            model = tf.keras.models.load_model(path)
        else:
            model = func()
            if save_weights:
                self.save_weights(model, f'{name}.h5')
        return model

    @staticmethod
    def save_weights(model, name):
        model.save(f'{name}.h5')

    @staticmethod
    def build_generator():
        down_stack = [
            downsample(64, 4, apply_batchnorm=False),
            downsample(128, 4),
            downsample(256, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4)
        ]
        up_stack = [
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4),
            upsample(256, 4),
            upsample(128, 4),
            upsample(64, 4)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                               kernel_initializer=initializer, activation='tanh')

        concat = tf.keras.layers.Concatenate()

        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        x = inputs

        # downsampling
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # upsampling and connecting
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    @staticmethod
    def build_discriminator(target=True, initial_units=64, layers=4, output_layer=None):
        return_target = False
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')

        if not target:
            x = inp
        else:
            target = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
            x = tf.keras.layers.concatenate([inp, target])
            return_target = True

        multipliyer = 1
        for layer in range(layers):
            if layer == 1:
                x = downsample(initial_units * multipliyer, 4, apply_batchnorm=False)(x)
                multipliyer *= 2
            else:
                x = downsample(initial_units * multipliyer, 4)(x)
                if multipliyer < 8:
                    multipliyer *= 2

        # TODO: Se han eliminado varias layers de zeropadding del discriminador pix2pix original

        if not output_layer:
            last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)
        else:
            last = output_layer

        if return_target:
            return tf.keras.Model(inputs=[inp, target], outputs=last)
        else:
            return tf.keras.Model(inputs=inp, outputs=last)

    # TODO: add this to the get_dataset method
    @staticmethod
    def set_input_shape(width, height):
        preprocessing.IMG_WIDTH = width
        preprocessing.IMG_HEIGHT = height

    @staticmethod
    def get_dataset(input_path, output_path, split=0.2, batch_size=1, file_shuffle=False):
        buffer_size = max(len(input_path), len(output_path))

        input_path = glob.glob(input_path + r'\*.png')
        output_path = glob.glob(output_path + r'\*.png')

        x_train, x_test, y_train, y_test = train_test_split(input_path, output_path, test_size=split)

        # train
        input_dataset = tf.data.Dataset.list_files(x_train, shuffle=file_shuffle)
        output_dataset = tf.data.Dataset.list_files(y_train, shuffle=file_shuffle)

        train_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
        train_dataset = train_dataset.map(preprocessing.load_images_train).shuffle(buffer_size).batch(batch_size)

        # validation
        input_dataset = tf.data.Dataset.list_files(x_test, shuffle=file_shuffle)
        output_dataset = tf.data.Dataset.list_files(y_test, shuffle=file_shuffle)

        test_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
        test_dataset = test_dataset.map(preprocessing.load_images_train).batch(batch_size)

        return train_dataset, test_dataset

    # Metrics
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, disc_generated_output, gen_output, target):
        g_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = g_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss

    # Training functions
    @tf.function
    def train_step(self, train_x, train_y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(train_x, training=True)

            disc_real_output = self.discriminator([train_x, train_y], training=True)
            disc_generated_output = self.discriminator([train_x, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, train_y)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        if self.log_dir is not None:
            self.train_loss(disc_loss)
            self.train_real_acc(tf.ones_like(disc_real_output), disc_real_output)
            self.train_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)

    def validate(self, test_in, test_out):
        gen_output = self.generator(test_in, training=False)

        disc_real_output = self.discriminator([test_in, test_out], training=False)
        disc_generated_output = self.discriminator([test_in, gen_output], training=False)

        gen_loss = self.generator_loss(disc_generated_output, gen_output, test_out)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        if self.log_dir is not None:
            self.val_loss(disc_loss)
            self.val_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)
            self.val_real_acc(tf.ones_like(disc_real_output), disc_real_output)

    def fit(self, train_ds, test_ds=None, epochs=5):
        for epoch in range(epochs):
            # Train
            for input_image, target in train_ds:
                self.train_step(input_image, target)
            # Validation
            if test_ds is not None:
                for input_image, target_image in test_ds:
                    self.validate(input_image, target_image)

            # Tensorboard
            if self.log_dir is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy real', self.train_real_acc.result(), step=epoch)
                    tf.summary.scalar('accuracy generated', self.train_gen_acc.result(), step=epoch)

                self._train_predict(train_ds, self.train_summary_writer, 'train', epoch)

                self.train_loss.reset_states()
                self.train_gen_acc.reset_states()
                self.train_real_acc.reset_states()

                if test_ds is not None:
                    with self.val_summary_writer.as_default():
                        tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
                        tf.summary.scalar('accuracy real', self.val_real_acc.result(), step=epoch)
                        tf.summary.scalar('accuracy generated', self.val_gen_acc.result(), step=epoch)

                    self._train_predict(test_ds, self.val_summary_writer, 'validation', epoch)

                    self.val_loss.reset_states()
                    self.val_gen_acc.reset_states()
                    self.val_real_acc.reset_states()

    def predict(self, dataset, log_path, samples):
        writer = self._set_logdir(log_path, 'predict')
        step = 0
        if samples == 'all':
            target = dataset
        else:
            target = dataset.take(samples)

        for x, y in target.take(samples):
            prediction = self.generator(x, training=False)

            stack = tf.stack([x, prediction, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('predictions', stack, step=step)

    def _train_predict(self, dataset, writer, name, step):
        for x, y in dataset.take(1):
            generated = self.generator(x, training=False)
            stack = tf.stack([x, generated, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image(name, stack, step=step, max_outputs=3)


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

    def get_dataset(self, temples, dataset_path, ruins_per_temple=2):
        if dataset_path is None:
            dataset_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\\'

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

        return train_dataset

    @staticmethod
    def get_single_dataset(ruins_path, temple_path, mode='train'):
        if mode == 'train':
            preprocessing_function = cp.load_images_train
        else:
            preprocessing_function = cp.load_images_test

        ruins_path_list = glob.glob(ruins_path + r'\*.png')
        temple_path_list = glob.glob(temple_path + r'\*.png')

        repetition = len(ruins_path_list) // len(temple_path_list)

        ruins_dataset = tf.data.Dataset.list_files(ruins_path_list, shuffle=False)
        temple_dataset = tf.data.Dataset.list_files(temple_path_list, shuffle=False)

        temple_dataset = temple_dataset.repeat(repetition)

        full_dataset = tf.data.Dataset.zip((ruins_dataset, temple_dataset))
        full_dataset = full_dataset.map(preprocessing_function).batch(1)

        return full_dataset


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


def working_test():
    log_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\logs\just_ruins'
    sample_model = CustomPix2Pix(log_dir=log_path)

    input_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\temples\temple_0'
    output_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\colors_temples\colors_temple_0'
    ds_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\sketchup\dataset\\'

    cp.RESIZE_FACTOR = 1.3

    train = sample_model.get_dataset(['temple_0', 'temple_1', 'temple_5'], ds_path)
    sample_model.fit(train, epochs=100)


if __name__ == '__main__':
    # TODO: Si bien esta es la implementación base "naive" (usando pix2pix para todo, sin introducir modificaciones)
    #  puede mejorarse convirtiendo esa segmentación y desegmentación en un CycleGAN. Mejorará también una vez se
    #  tengan todos los templos con sus respectivos colores.
    other_gen = Pix2Pix.build_generator()
