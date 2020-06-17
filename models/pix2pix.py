"""
Title: Pix2Pix
Author: [AMSUFV](https://github.com/AMSUFV)
Date created:
Last modified:
Description: Pix2Pix implementation
"""

"""
## Setup
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Create the preprocessing functions
"""


def load_train(image_path):
    stack = load(image_path)
    stack = random_jitter(stack)
    stack = normalize(stack)
    stack = tf.unstack(stack, num=stack.shape[0])

    return stack


def load_test(image_path):
    stack = load(image_path)
    stack = resize(stack, HEIGHT, WIDTH)
    stack = normalize(stack)
    stack = tf.unstack(stack, num=stack.shape[0])

    return stack


def load(img_path):
    file = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(file)

    # images contain both input and target image
    width = tf.shape(image)[1]
    middle = width // 2

    real_image = image[:, :middle, :]
    segmented_image = image[:, middle:, :]

    # images are stacked for better handling
    stack = tf.stack([real_image, segmented_image])
    stack = tf.cast(stack, tf.float32)

    return stack


def resize(stack, height, width):
    return tf.image.resize(
        stack, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


def normalize(stack):
    return (stack / 127.5) - 1


def random_jitter(stack):
    # random jitter is applied by upscaling to 110% the size
    stack = resize(stack, int(WIDTH * 1.1), int(HEIGHT * 1.1))
    # cropping randomnly back to the desired size
    stack = tf.image.random_crop(stack, size=[stack.shape[0], HEIGHT, WIDTH, 3])
    # and performing random mirroring
    if tf.random.uniform(()) > 0.5:
        return tf.image.flip_left_right(stack)
    else:
        return stack


"""
## Downscale and upscale blocks
"""


class Downscale(layers.Layer):
    def __init__(self, filters, size, apply_norm=True, slope=0.2):
        super(Downscale, self).__init__()
        self.apply_norm = apply_norm
        self.slope = slope
        w_init = tf.random_normal_initializer(0.0, 0.02)

        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=w_init,
            use_bias=False,
        )
        if apply_norm:
            self.batch_norm = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        if self.apply_norm:
            x = self.batch_norm(x)
        return tf.nn.leaky_relu(x, alpha=self.slope)


class Upscale(layers.Layer):
    def __init__(self, filters, size, apply_dropout=False, rate=0.5):
        super(Upscale, self).__init__()
        self.apply_droput = apply_dropout
        self.rate = rate
        w_init = tf.random_normal_initializer(0.0, 0.02)
        self.t_conv = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=w_init,
            use_bias=False,
        )
        self.batch_norm = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.t_conv(inputs)
        x = self.batch_norm(x)
        if self.apply_droput:
            x = tf.nn.dropout(x, rate=self.rate)
        return tf.nn.relu(x)

"""
## Build the generator 
"""


def build_generator():
    initializer = tf.random_normal_initializer(0.0, 0.02)
    input_image = x = layers.Input(shape=(None, None, 3))

    down_stack = [
        Downscale(64, 4, apply_norm=False),
        Downscale(128, 4),
        Downscale(256, 4),
        Downscale(512, 4),
        Downscale(512, 4),
        Downscale(512, 4),
        Downscale(512, 4),
        Downscale(512, 4),
    ]
    up_stack = [
        Upscale(512, 4, apply_dropout=True),
        Upscale(512, 4, apply_dropout=True),
        Upscale(512, 4, apply_dropout=True),
        Upscale(512, 4),
        Upscale(256, 4),
        Upscale(128, 4),
        Upscale(64, 4),
    ]

    skips = []
    for block in down_stack:
        x = block(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    output_image = layers.Conv2DTranspose(
        filters=3,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )(x)

    return keras.Model(
        inputs=input_image, outputs=output_image, name="pix2pix_generator"
    )


"""
## Build the discriminator 
"""


def build_discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    input_image = layers.Input(shape=(None, None, 3), name="input_image")
    target_image = layers.Input(shape=(None, None, 3), name="target_image")
    x = layers.Concatenate()([input_image, target_image])

    x = Downscale(64, 4, apply_norm=False)(x)
    x = Downscale(128, 4)(x)
    x = Downscale(256, 4)(x)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(
        filters=512,
        kernel_size=4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.ZeroPadding2D()(x)

    markov_rf = layers.Conv2D(
        filters=1, kernel_size=4, strides=1, kernel_initializer=initializer
    )(x)

    return keras.Model(inputs=[input_image, target_image], outputs=markov_rf)


"""
Define the losses
"""
bce = keras.losses.BinaryCrossentropy(from_logits=True)


def loss_g(y, gx, dgx):
    loss_dgx = bce(tf.ones_like(dgx), dgx)
    loss_l1 = tf.reduce_mean(tf.abs(y - gx))
    total_loss = loss_dgx + LAMBDA * loss_l1
    return total_loss, loss_l1


def loss_d(dy, dgx):
    loss_y = bce(tf.ones_like(dy), dy)
    loss_gx = bce(tf.zeros_like(dgx), dgx)
    return (loss_y + loss_gx) / 2


"""
## Define Pix2Pix as a `Model` with a custom `train_step`
"""


class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super(Pix2Pix, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def train_g(self, x, y):
        with tf.GradientTape() as t:
            gx = self.generator(x, training=True)
            dgx = self.discriminator([x, gx], training=True)
            g_loss, l1_loss = self.g_loss_fn(y, gx, dgx)

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )

        return gx, g_loss, l1_loss

    def train_d(self, x, gx, y):
        with tf.GradientTape() as t:
            dy = self.discriminator([x, y], training=True)
            dgx = self.discriminator([x, gx], training=True)
            d_loss = self.d_loss_fn(dy, dgx)

        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_variables)
        )

        return d_loss

    def train_step(self, images):
        x, y = images
        gx, g_loss, l1_loss = self.train_g(x, y)
        d_loss = self.train_d(x, gx, y)
        return {"d_loss": d_loss, "g_loss": g_loss, "l1_loss": l1_loss}


"""
## Prepare the dataset
"""

# Variables
BUFFER_SIZE = 400
BATCH_SIZE = 1
LAMBDA = 100
WIDTH = HEIGHT = 256

url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"
path = keras.utils.get_file("facades.tar.gz", origin=url, extract=True)
path = os.path.join(os.path.dirname(path), "facades/")

train = tf.data.Dataset.list_files(path + "train/*.jpg")
train = train.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


"""
## Train the end-to-end model
"""

generator = build_generator()
discriminator = build_discriminator()
generator.summary()
discriminator.summary()


pix2pix = Pix2Pix(generator, discriminator)
pix2pix.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
    d_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
    g_loss_fn=loss_g,
    d_loss_fn=loss_d,
)

pix2pix.fit(train, epochs=5)
