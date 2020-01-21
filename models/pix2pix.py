from __future__ import absolute_import, division, print_function, unicode_literals

import datetime

import tensorflow as tf
import time
# creación del dataset
import glob
from itertools import compress
from sklearn.model_selection import train_test_split

# modelo base y preprocesamiento
from models.utils import pix2pix_preprocessing as preprocessing
from models.utils.basemodel import BaseModel
from models.utils.metric_logger import MetricLogger


class Pix2Pix(BaseModel):
    def __init__(self, *, gen_path=None, disc_path=None, log_dir='logs'):
        self.generator, self.discriminator = self.set_weights(gen_path, disc_path)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.LAMBDA = 100

        # TODO: Single responsibility; sacar las métricas
        # if keep_logs:
        #     self.metric_logger = MetricLogger(log_dir=log_dir, default_metrics=True)
        # else:
        #     self.metric_logger = None
        # self.metric_logger = MetricLogger(r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\logs\pix2pix')

        # Tensorboard
        # log_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\logs\pix2pix'
        self.train_summary_writer, self.val_summary_writer = self.set_logdirs(log_dir)
        # Métricas
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_real_acc = tf.keras.metrics.BinaryAccuracy('train_real_accuracy')
        self.train_gen_acc = tf.keras.metrics.BinaryAccuracy('train_gen_accuracy')

        self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.val_gen_acc = tf.keras.metrics.BinaryAccuracy('val_gen_accuracy')
        self.val_real_acc = tf.keras.metrics.BinaryAccuracy('val_real_accuracy')

    @staticmethod
    def set_logdirs(path):
        """
        Establece dónde se guardarán los logs del entrenamiento

        :param path: String. Path a la carpeta donde se guardarán los logs del entrenamiento
        :return: file writer, file writer
        """
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = path + r'\\' + current_time + r'\train'
        val_log_dir = path + r'\\' + current_time + r'\val'

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        return train_summary_writer, val_summary_writer

    # Creación de la red
    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                          kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    @staticmethod
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

    def build_generator(self):
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),
            self.upsample(64, 4),
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

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, apply_batchnorm=False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    # Parent class implementation
    def set_weights(self, gen_path, disc_path):
        if gen_path is not None and disc_path is not None:
            generator = tf.keras.models.load_model(gen_path)
            discriminator = tf.keras.models.load_model(disc_path)
        else:
            generator = self.build_generator()
            discriminator = self.build_discriminator()
            self.save_weights(generator, 'initial_generator.h5')
            self.save_weights(discriminator, 'initial_discriminator.h5')
        return generator, discriminator

    @staticmethod
    def save_weights(model, name):
        model.save(name)

    @staticmethod
    def set_input_shape(width, height):
        preprocessing.IMG_WIDTH = width
        preprocessing.IMG_HEIGHT = height

    @staticmethod
    def get_dataset(input_path, real_path, split=0.2, file_shuffle=True):
        """Método genérico de creación de datasets

        :param input_path: ruta a la carpeta con los inputs de la red
        :param real_path: ruta a la carpeta con los outputs esperados
        :param split: porcentaje de division train/test
        :param file_shuffle: barajar o no los archivos; igualar a  False si inputs y outputs guardan relación directa
        :return: train_dataset, test_dataset
        """
        buffer_size = min(len(input_path), len(real_path))
        batch_size = 1

        input_path = glob.glob(input_path + r'\*.png')
        real_path = glob.glob(real_path + r'\*.png')
        input_path = input_path[:buffer_size]
        real_path = real_path[:buffer_size]

        x_train, x_test, y_train, y_test = train_test_split(input_path, real_path, test_size=split)

        # train
        input_dataset = tf.data.Dataset.list_files(x_train, shuffle=file_shuffle)
        output_dataset = tf.data.Dataset.list_files(y_train, shuffle=file_shuffle)

        train_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
        train_dataset = train_dataset.map(preprocessing.load_images_train).shuffle(buffer_size).batch(batch_size)

        input_dataset = tf.data.Dataset.list_files(x_test, shuffle=False)
        output_dataset = tf.data.Dataset.list_files(y_test, shuffle=False)

        test_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
        test_dataset = test_dataset.map(preprocessing.load_images_train).batch(batch_size)

        return train_dataset, test_dataset

    # Metrics
    # class metrics:
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        g_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
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

        # Tensorboard
        # if self.metric_logger is not None:
        #     self.metric_logger.update_metric('train', 'loss', disc_loss)
        #     self.metric_logger.update_metric('train', 'accuracy')

        self.train_loss(disc_loss)
        self.train_real_acc(tf.ones_like(disc_real_output), disc_real_output)
        self.train_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)

    def validate(self, test_in, test_out):
        gen_output = self.generator(test_in, training=False)

        disc_real_output = self.discriminator([test_in, test_out], training=False)
        disc_generated_output = self.discriminator([test_in, gen_output], training=False)

        gen_loss = self.generator_loss(disc_generated_output, gen_output, test_out)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        self.val_loss(disc_loss)
        self.val_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)
        self.val_real_acc(tf.ones_like(disc_real_output), disc_real_output)

    def fit(self, train_ds, test_ds, epochs, save_path=None):
        for epoch in range(epochs):
            # Train
            for input_image, target in train_ds:
                self.train_step(input_image, target)
            # # Validation
            for input_image, target_image in test_ds:
                self.validate(input_image, target_image)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy real', self.train_real_acc.result(), step=epoch)
                tf.summary.scalar('accuracy generated', self.train_gen_acc.result(), step=epoch)
                # images
                stack = self.get_tb_stack(train_ds)
                tf.summary.image('train', stack, step=epoch)

            with self.val_summary_writer.as_default():
                tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
                tf.summary.scalar('accuracy real', self.val_real_acc.result(), step=epoch)
                tf.summary.scalar('accuracy generated', self.val_gen_acc.result(), step=epoch)

                # images
                stack = self.get_tb_stack(test_ds)
                tf.summary.image('validation', stack, step=epoch)
            #
            # # Tensorboard
            # self.metric_logger.write_metrics('train', epoch)
            # self.metric_logger.write_metrics('validation', epoch)
            #
            # self.metric_logger.reset_metrics('train')
            # self.metric_logger.reset_metrics('validation')

            # self.train_loss.reset_states()
            # self.train_gen_acc.reset_states()
            # self.train_real_acc.reset_states()
            #
            # self.val_loss.reset_states()
            # self.val_gen_acc.reset_states()
            # self.val_real_acc.reset_states()

    def get_tb_stack(self, dataset):
        for x, y in dataset.take(1):
            # x is the input
            # y is the ground truth
            generated = self.generator(x, training=False)
            stack = tf.stack([x, generated, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            return stack

    # TODO: Revisit this
    def predict(self, path, save_path=None):
        pass


class CustomPix2Pix(Pix2Pix):
    def get_complete_datset(self, temples, ruins_per_temple=1):
        """
        Este método asume una estructura de archivos en la que los templos completos están en una carpeta llamada
        temples y llamados temple_0, temple_1, etc, y sus ruinas en la carpeta temples_ruins
        :param temples:
        :param ruins_per_temple:
        :return:
        """
        for i, temple in enumerate(temples):
            output_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\temples\\' + temple
            input_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\temples_ruins\\' + temple
            if i == 0:
                train_dataset, val_dataset = self.get_dataset(input_path, output_path, ruins_per_temple)
            else:
                tr, val = self.get_dataset(input_path, output_path, ruins_per_temple)
                train_dataset = train_dataset.concatenate(tr)
                val_dataset = val_dataset.concatenate(val)

        return train_dataset, val_dataset

    # dataset creation function
    @staticmethod
    def get_dataset(input_path, real_path, repeat_real=1):
        """Generación del dataset. Orientado a la extracción de diferentes ángulos de templos griegos

        :param input_path: ruta a las imágenes de ruinas de templos
        :param real_path: ruta a las imágenes de templos completos
        :param repeat_real: el número de veces que las imágenes de templos completos se repiten; tantas como diferentes
                            modelos de sus ruinas se tengan
        :return: train_dataset, test_datset
        """
        buffer_size = len(input_path)
        batch_size = 1

        input_path = glob.glob(input_path + r'*\*.png')
        real_path = glob.glob(real_path + r'\*.png')

        test_mask = ([False] * (len(real_path) // 100 * 8) + [True] * (len(real_path) // 100 * 2)) * 10
        train_mask = ([True] * (len(real_path) // 100 * 8) + [False] * (len(real_path) // 100 * 2)) * 10

        train_input = list(compress(input_path, train_mask * repeat_real))
        train_real = list(compress(real_path, train_mask))

        test_input = list(compress(input_path, test_mask * repeat_real))
        test_real = list(compress(real_path, test_mask))

        # train
        input_dataset = tf.data.Dataset.list_files(train_input, shuffle=False)
        real_datset = tf.data.Dataset.list_files(train_real, shuffle=False)
        real_datset = real_datset.repeat(repeat_real)

        train_dataset = tf.data.Dataset.zip((input_dataset, real_datset))
        train_dataset = train_dataset.map(preprocessing.load_images_train).shuffle(buffer_size).batch(batch_size)

        # test
        test_input_ds = tf.data.Dataset.list_files(test_input, shuffle=False)
        test_real_ds = tf.data.Dataset.list_files(test_real, shuffle=False)
        test_real_ds = test_real_ds.repeat(repeat_real)

        test_dataset = tf.data.Dataset.zip((test_input_ds, test_real_ds))
        test_dataset = test_dataset.map(preprocessing.load_images_test).batch(batch_size)

        return train_dataset, test_dataset


class StylePix2Pix(Pix2Pix):
    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')

        down1 = self.downsample(64, 4, apply_batchnorm=False)(inp)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)

        norm = tf.keras.layers.BatchNormalization()(conv)
        # norm = InstanceNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(norm)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=inp, outputs=last)

    def generator_loss(self, disc_generated_output):
        return self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    @tf.function
    def train_step(self, train_x, train_y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(train_x, training=True)

            disc_real_output = self.discriminator(train_y, training=True)
            disc_generated_output = self.discriminator(gen_output, training=True)

            gen_loss = self.generator_loss(disc_generated_output)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        # Tensorboard
        self.train_loss(disc_loss)

    def validate(self, test_in, test_out):
        gen_output = self.generator(test_in, training=False)

        disc_real_output = self.discriminator(test_out, training=False)
        disc_generated_output = self.discriminator(gen_output, training=False)

        gen_loss = self.generator_loss(disc_generated_output)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        self.val_loss(disc_loss)


if __name__ == '__main__':
    pix2pix = CustomPix2Pix(log_dir=r'logs\\custom_pix2pix')
    # preprocessing.RESIZE_FACTOR = 3
    # train, test = pix2pix.get_dataset(r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\temples\temple_0',
    #                                   r'C:\Users\Ceiec06\Documents\GitHub\CEIEC-GANs\greek_temples_dataset\Colores')

    # train, test = pix2pix.get_dataset(r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\ruins\temple_0',
    #                                   r'C:\Users\Ceiec06\Documents\GitHub\CEIEC-GANs\greek_temples_dataset\restored_png')

    # train, test = pix2pix.get_dataset(r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\temples_ruins\temple_0_ruins_0',
    #                                   r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\temples\temple_0')
    train, test = pix2pix.get_complete_datset(temples=['temple_0'], ruins_per_temple=2)
    pix2pix.fit(train, test, 100)
