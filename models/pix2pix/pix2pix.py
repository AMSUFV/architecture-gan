from __future__ import absolute_import, division, print_function, unicode_literals

import datetime

import tensorflow as tf
import time
# creación del dataset
import glob
from itertools import compress
from sklearn.model_selection import train_test_split
# imágenes
import matplotlib.pyplot as plt
from IPython.display import clear_output
# modelo base y preprocesamiento
from models.basemodel import BaseModel
from models.pix2pix import pix2pix_preprocessing as preprocessing


class Pix2Pix(BaseModel):
    def __init__(self, *, gen_path=None, disc_path=None):
        self.generator, self.discriminator = self.set_weights(gen_path, disc_path)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.LAMBDA = 100

        # TODO: Integración con Tensorboard
        # Tensorboard
        # Métricas
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        # Directorio
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        # Train
        self.train_log_dir = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\logs\gradient_tape\\' + current_time + r'\train'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        # Validation
        self.val_log_dir = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\logs\gradient_tape\\' + current_time + r'\val'
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

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

    def generator(self):
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

    def discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

        # x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, apply_batchnorm=False)(inp)  # (bs, 128, 128, 64)
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

        return tf.keras.Model(inputs=inp, outputs=last)

    # Implementación de los métodos de la clase padre
    def set_weights(self, gen_path, disc_path):
        if gen_path is not None and disc_path is not None:
            generator = tf.keras.models.load_model(gen_path)
            discriminator = tf.keras.models.load_model(disc_path)
        else:
            generator = self.generator()
            discriminator = self.discriminator()
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

    # dataset creation function
    @staticmethod
    def get_dataset(input_path, real_path, repeat_real=1):
        """Generación del dataset. Orientado a la extracción de diferentes ángulos de templos griegos

        :param input_path: ruta a las imágenes de ruinas de templos
        :param real_path: ruta a las imágenes de templos completos
        :param repeat_real: el número de veces que las imágenes de templos completos se repiten; tantas
        como diferentes modelos de sus ruinas se tengan
        :return: train_dataset, test_datset
        """
        buffer_size = len(input_path)
        batch_size = 1

        input_path = glob.glob(input_path + r'\*.png')
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

    # Metrics
    # class metrics:
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss

    # Training functions
    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
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

        disc_real_output = self.discriminator([test_in, test_out], training=False)
        disc_generated_output = self.discriminator([test_in, gen_output], training=False)

        gen_loss = self.generator_loss(disc_generated_output, gen_output, test_out)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        self.val_loss(disc_loss)

    def fit(self, train_ds, test_ds, epochs, save_path=None):
        for epoch in range(epochs):
            start = time.time()

            # Train
            for input_image, target in train_ds:
                self.train_step(input_image, target)

            for input_image, target_image in test_ds:
                self.validate(input_image, target_image)

            # Tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

            with self.val_summary_writer.as_default():
                tf.summary.scalar('loss', self.val_loss.result(), step=epoch)

            self.train_loss.reset_states()
            self.val_loss.reset_states()

            # Validation
            # TODO: Finish implementing validation

            print('Time taken for epoch {} is {:.2f} sec\n'.format(epoch + 1, time.time() - start))
            # TODO: Image generation and integration with tensorflow (predict)

    def predict(self, path, save_path=None):
        image = preprocessing.load_single_image(path)
        prediction = self.generator(image, training=False)
        if save_path is not None:
            clear_output(wait=True)
            plt.figure()
            plt.imshow(prediction[0] * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig(f'{save_path}/image.png', pad_inches=0, bbox_inches='tight')
            plt.close()
        else:
            return prediction


class StyleTransfer(Pix2Pix):
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

        input_path = glob.glob(input_path + '*')
        real_path = glob.glob(real_path + '*')

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


if __name__ == '__main__':
    pix2pix = Pix2Pix()
    train, test = pix2pix.get_dataset(r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\ruins\temple_0',
                                      r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\ruins\temple_0_ruins_0')
    pix2pix.fit(train, test, 20)
