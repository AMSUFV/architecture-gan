#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

import glob
from random import shuffle

from itertools import compress


class Pix2Pix:
    def __init__(self, img_width, img_height, path_inp, path_real, epochs):
        self.img_width = img_width
        self.img_height = img_height
        self.input_path = path_inp
        self.real_path = path_real
        self.epochs = epochs        
        
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.generator = self.generator()
        self.discriminator = self.discriminator()
        
#   image processing related functions
    class image_processing:
        def load(input_path, real_path):

            paths = [input_path, real_path]
            images = []

            for path in paths:
                # Loading the image
                image = tf.io.read_file(path)
                image = tf.image.decode_png(image)
                image = tf.cast(image, tf.float32)

                images.append(image)

            return images[0], images[1]

        def resize(input_image, real_image, height, width):
            input_image = tf.image.resize(input_image, [height, width],
                                         method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            real_image = tf.image.resize(real_image, [height, width],
                                         method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return input_image, real_image

        def normalize(input_image, real_image):
            input_image = (input_image / 127.5) - 1
            real_image = (real_image / 127.5) - 1

            return input_image, real_image

        def random_crop(input_image, real_image):
            stacked_image = tf.stack([input_image, real_image], axis=0)
            cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

            return cropped_image[0], cropped_image[1]

        @tf.function()
        def random_jitter(input_image, real_image):
        #     En el codigo original hacen un resize al 111% del tamaño, 256 a 286, se hace lo propio
        #     con las dimensiones de nuestra imagen
            resized_width = IMG_WIDTH + IMG_WIDTH // 10
            resized_height = IMG_HEIGHT +  IMG_HEIGHT// 10
        #     Resize
            input_image, real_image = resize(input_image, real_image, resized_height, resized_width)
        #     Random crop
            input_image, real_image = random_crop(input_image, real_image)

            if tf.random.uniform(()) > 0.5:
        #         random mirroring
                input_image = tf.image.flip_left_right(input_image)
                real_image = tf.image.flip_left_right(real_image)

            return input_image, real_image

        def load_images_train(input_path, real_path):
            input_image, real_image = load(input_path, real_path)
            input_image, real_image = random_jitter(input_image, real_image)
            input_image, real_image = normalize(input_image, real_image)

            return input_image, real_image

        def load_images_test(input_path, real_path):
            input_image, real_image = load(input_path, real_path)
            input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
            input_image, real_image = normalize(input_image, real_image)

            return input_image, real_image

        
    # dataset creation function
    def gen_dataset(self, input_path, real_path, repeat_real = 1):
        test_mask = ([False] * (len(real_path)//100 * 8) + [True] * (len(real_path)//100 * 2)) * 10
        train_mask = ([True] * (len(real_path)//100 * 8) + [False] * (len(real_path)//100 * 2)) * 10
        
        train_input = list(compress(input_path, train_mask * repeat_real))
        train_real = list(compress(real_path, train_mask))

        test_input = list(compress(input_path, test_mask * repeat_real))
        test_real = list(compress(real_path, test_mask))
        
        # train
        input_dataset = tf.data.Dataset.list_files(train_input, shuffle=False)
        real_datset = tf.data.Dataset.list_files(train_real, shuffle=False)
        real_datset = real_datset.repeat(repeat_real)

        train_dataset = tf.data.Dataset.zip((input_dataset, real_datset))
        train_dataset = train_dataset.map(load_images_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        
        # test
        test_input_ds = tf.data.Dataset.list_files(test_input, shuffle = False)
        test_real_ds = tf.data.Dataset.list_files(test_real, shuffle = False)
        test_real_ds = test_real_ds.repeat(repeat_real)

        test_dataset = tf.data.Dataset.zip((test_input_ds, test_real_ds))
        test_dataset = test_dataset.map(load_images_train).batch(BATCH_SIZE)
        
        return train_dataset, test_dataset
    
    
    # net-creating functions
    class buildingblocks:
        def downsample(filters, size, apply_batchnorm = True):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                              kernel_initializer = initializer, use_bias=False))

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
    
    def generator(self):
        down_stack = [
            # (256, 512)
            self.buildingblocks.downsample(64, 4, apply_batchnorm=False), # 128, 256
            self.buildingblocks.downsample(128, 4), # 64, 128
            self.buildingblocks.downsample(256, 4), # 32, 64
            self.buildingblocks.downsample(512, 4), # 16, 32
            self.buildingblocks.downsample(512, 4), # 8, 16
            self.buildingblocks.downsample(512, 4), # 4, 8
            self.buildingblocks.downsample(512, 4), # 2, 4
            self.buildingblocks.downsample(512, 4)
        ]

        up_stack = [
            self.buildingblocks.upsample(512, 4, apply_dropout=True), 
            self.buildingblocks.upsample(512, 4, apply_dropout=True), 
            self.buildingblocks.upsample(512, 4, apply_dropout=True), 
            self.buildingblocks.upsample(512, 4), 
            self.buildingblocks.upsample(256, 4), 
            self.buildingblocks.upsample(128, 4), 
            self.buildingblocks.upsample(64, 4), 
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

        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

        down1 = self.buildingblocks.downsample(64, 4, apply_batchnorm=False)(inp) # (bs, 128, 128, 64)
        down2 = self.buildingblocks.downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = self.buildingblocks.downsample(256, 4)(down2) # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=inp, outputs=last)
    
    
    # Training functions
    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

