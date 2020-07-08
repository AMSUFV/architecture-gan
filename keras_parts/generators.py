import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .blocks import Downscale, Upscale


def pix2pix_generator(input_shape=(None, None, 3), assisted=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    if assisted:
        input_layer = [layers.Input(shape=input_shape), layers.Input(shape=input_shape)]
        x = layers.concatenate(input_layer)
    else:
        input_layer = x = layers.Input(shape=input_shape)

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
        x = layers.concatenate([x, skip])

    output_image = layers.Conv2DTranspose(
        filters=3,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )(x)

    return keras.Model(
        inputs=input_layer, outputs=output_image, name="pix2pix_generator"
    )
