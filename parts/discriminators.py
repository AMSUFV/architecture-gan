import tensorflow as tf
from parts.blocks import downsample


def pix2pix(input_shape=None, initial_units=64, layers=4, norm_type='batchnorm'):
    if input_shape is None:
        input_shape = [None, None, 3]

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    target_image = tf.keras.layers.Input(shape=input_shape, name='target_image')
    x = tf.keras.layers.concatenate([inp, target_image])

    multiplier = 1
    for layer in range(layers):
        if layer == 1:
            x = downsample(x, filters=initial_units * multiplier, kernel_size=4, apply_norm=False)
            multiplier *= 2
        else:
            x = downsample(x, filters=initial_units * multiplier, kernel_size=4, apply_norm=False)
            if multiplier < 8:
                multiplier *= 2

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, activation='sigmoid')(x)

    return tf.keras.Model(inputs=[inp, target_image], outputs=last)
