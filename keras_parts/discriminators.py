import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from .blocks import downscale


def pix2pix_discriminator(input_shape=(None, None, 3), norm_type="batch"):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    input_image = layers.Input(shape=input_shape, name="input_image")
    target_image = layers.Input(shape=input_shape, name="target_image")
    x = layers.concatenate([input_image, target_image])

    x = downscale(x, 64, 4, apply_norm=False)
    x = downscale(x, 128, 4, norm_type=norm_type)
    x = downscale(x, 256, 4, norm_type=norm_type)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(
        filters=512,
        kernel_size=4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False,
    )(x)

    if norm_type == "batch":
        x = layers.BatchNormalization()(x)
    elif norm_type == "instance":
        x = tfa.layers.InstanceNormalization()(x)
    else:
        raise Exception(f"Norm type not recognized: {norm_type}")

    x = layers.LeakyReLU()(x)
    x = layers.ZeroPadding2D()(x)

    markov_rf = layers.Conv2D(
        filters=1, kernel_size=4, strides=1, kernel_initializer=initializer
    )(x)

    return keras.Model(inputs=[input_image, target_image], outputs=markov_rf)
