import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .blocks import downscale, upscale


def pix2pix_generator(input_shape=(None, None, 3), assisted=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    if assisted:
        input_layer = [layers.Input(shape=input_shape), layers.Input(shape=input_shape)]
        x = layers.concatenate(input_layer)
    else:
        input_layer = x = layers.Input(shape=input_shape)

    down_stack = [
        dict(filters=64, kernel_size=4, apply_norm=False),
        dict(filters=128, kernel_size=4),
        dict(filters=256, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
        dict(filters=512, kernel_size=4),
    ]
    up_stack = [
        dict(filters=512, kernel_size=4, apply_dropout=True),
        dict(filters=512, kernel_size=4, apply_dropout=True),
        dict(filters=512, kernel_size=4, apply_dropout=True),
        dict(filters=512, kernel_size=4),
        dict(filters=256, kernel_size=4),
        dict(filters=128, kernel_size=4),
        dict(filters=64, kernel_size=4),
    ]

    skips = []
    for block in down_stack:
        x = downscale(
            x,
            block.get('filters'),
            block.get('kernel_size'),
            block.get('apply_norm'),
        )
        skips.append(x)

    skips = reversed(skips[:-1])
    for block, skip in zip(up_stack, skips):
        x = upscale(
            x,
            block.get('filters'),
            block.get('kernel_size'),
            block.get('apply_dropout'),
        )
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
