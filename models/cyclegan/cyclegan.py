from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

from models.pix2pix import pix2pix_preprocessing as preprocessing


class CycleGAN:
    def __init__(self, *, img_width=512, img_height=256, epochs=200, save_path=None):
        preprocessing.IMG_WIDTH = img_width
        preprocessing.IMG_HEIGHT = img_height

        self.epochs = epochs
        self.LAMBDA = 10

        self.generator_xy_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_yx_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_xy = self.generator()
        self.generator_yx = self.generator()

        self.discriminator_x = self.discriminator()
        self.discriminator_y = self.discriminator()

        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # metricas
        

    # dataset creation function
    @staticmethod
    def gen_dataset(*, paths_x, paths_y):
        # train-test 0.25 split
        train_x, test_x = train_test_split(paths_x)
        train_y, test_y = train_test_split(paths_y)

        train_x = tf.data.Dataset.list_files(train_x)
        train_y = tf.data.Dataset.list_files(train_y)
        train_xy = tf.data.Dataset.zip((train_x, train_y))

        test_x = tf.data.Dataset.list_files(test_x)
        test_y = tf.data.Dataset.list_files(test_y)
        test_xy = tf.data.Dataset.zip((test_x, test_y))

        train_xy = train_xy.map(preprocessing.load_images_train).batch(1)
        test_xy = test_xy.map(preprocessing.load_images_test).batch(1)

        return train_xy, test_xy

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
            # (256, 512)
            self.downsample(64, 4, apply_batchnorm=False),  # 128, 256
            self.downsample(128, 4),  # 64, 128
            self.downsample(256, 4),  # 32, 64
            self.downsample(512, 4),  # 16, 32
            self.downsample(512, 4),  # 8, 16
            self.downsample(512, 4),  # 4, 8
            self.downsample(512, 4),  # 2, 4
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

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

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

    # Metricas
    def discriminator_loss(self, real, generated):
        real_loss = self.loss_object(tf.ones_like(real), real)
        generated_loss = self.loss_object(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_object(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    # Funciones de entrenamiento
    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            # x to y and back
            fake_y = self.generator_xy(real_x, training=True)
            cycled_x = self.generator_yx(fake_y, training=True)
            # y to x and back
            fake_x = self.generator_yx(real_y, training=True)
            cycled_y = self.generator_xy(fake_x, training=True)

            # x and y through x and y generators to calculate identity loss
            same_y = self.generator_xy(real_y, training=True)
            same_x = self.generator_yx(real_x, training=True)

            # discriminator output for real images
            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)
            # discriminator output for generated images
            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculating the loss
            gen_xy_loss = self.generator_loss(disc_fake_y)
            gen_yx_loss = self.generator_loss(disc_fake_x)

            # cycle loss
            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # total generator loss, adversarial loss plus cycle loss
            total_gen_xy_loss = gen_xy_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_yx_loss = gen_yx_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # gradients
        gen_xy_gradients = tape.gradient(total_gen_xy_loss, self.generator_xy.trainable_variables)
        gen_yx_gradients = tape.gradient(total_gen_yx_loss, self.generator_yx.trainable_variables)

        disc_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        disc_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        # applying the gradients
        self.generator_xy_optimizer.apply_gradients(zip(gen_xy_gradients, self.generator_xy.trainable_variables))
        self.generator_yx_optimizer.apply_gradients(zip(gen_yx_gradients, self.generator_yx.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(disc_x_gradients, self.discriminator_x.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(zip(disc_y_gradients, self.discriminator_y.trainable_variables))

    def fit(self, train_ds, test_ds):
        # Se toman imagenes para apreciar la evolucion del modelo
        for (_, _), (test_x, test_y) in zip(train_ds.take(1), test_ds.take(1)):
            plot_test_x = test_x
            plot_test_y = test_y

        for epoch in range(self.epochs):
            start = time.time()
            # Train
            for image_x, image_y in train_ds:
                self.train_step(image_x, image_y)

            clear_output(wait=True)
            print('Time taken for epoch {}: {:.2f}'.format(epoch+1, time.time() - start))

            # Imagenes
            if (epoch + 1) % 2 == 0 or epoch == 0 and self.save_path is not None:
                self.generate_images(self.generator_xy, plot_test_x, f'{epoch}xy')
                self.generate_images(self.generator_yx, plot_test_y, f'{epoch}yx')

    # Generación de imágenes
    def generate_images(self, model, image, name):
        prediction = model(image, training=False)

        display_list = [image[0], prediction[0]]
        title = ['Input image', 'Styled image']

        plt.close()
        plt.figure(figsize=(12, 12))
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.savefig(f'{self.save_path}/{name}.png', pad_inches=None, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    cyclegan = CycleGAN(save_path='C:/Users/Ceiec06/Documents/GitHub/ARQGAN/test/')

    paths_fakes = glob.glob('C:/Users/Ceiec06/Documents/GitHub/CEIEC-GANs/greek_temples_dataset/ruinas/*.png')
    paths_reals = glob.glob('C:/Users/Ceiec06/Documents/GitHub/ARQGAN/ruins_00/*.png')

    train, test = cyclegan.gen_dataset(paths_x=paths_reals, paths_y=paths_fakes)
    cyclegan.fit(train, test)
