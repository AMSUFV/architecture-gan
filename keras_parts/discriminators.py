import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .blocks import Downscale


def build_discriminator(input_shape=(None, None, 3)):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    input_image = layers.Input(shape=input_shape, name="input_image")
    target_image = layers.Input(shape=input_shape, name="target_image")
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
