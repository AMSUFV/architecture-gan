import tensorflow as tf
import tensorflow_addons as tfa


# TODO add comments
def downsample(x, filters, kernel_size, strides=2, apply_norm=True, norm_type='batchnorm'):
    initializer = tf.random_normal_initializer(0., 0.02)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='same', kernel_initializer=initializer, use_bias=False)(x)
    if apply_norm:
        if norm_type == 'batchnorm':
            x = tf.keras.layers.BatchNormalization()(x)
        elif norm_type == 'instancenorm':
            x = tfa.layers.InstanceNormalization()(x)
        else:
            raise Exception(f'{norm_type} is not supported.')

    x = tf.keras.layers.LeakyReLU()(x)

    return x


def upsample(x, filters, kernel_size, strides=2, apply_dropout=False, norm_type='batchnorm'):
    initializer = tf.random_normal_initializer(0., 0.02)

    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding='same', kernel_initializer=initializer, use_bias=False)(x)

    if norm_type == 'batchnorm':
        x = tf.keras.layers.BatchNormalization()(x)
    elif norm_type == 'instancenorm':
        x = tfa.layers.InstanceNormalization()(x)
    else:
        raise Exception(f'{norm_type} is not supported.')

    if apply_dropout:
        x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.ReLU()(x)

    return x
