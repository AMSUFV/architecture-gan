import tensorflow as tf
import tensorflow_addons as tfa


# TODO: add reflection padding
def resnet():
    """CycleGAN implementation https://arxiv.org/abs/1703.10593

    Per the paper's appendix:

    Generator

    'Let c7s1-k denote a 7×7 Convolution-InstanceNormReLU layer with k filters and stride 1. dk denotes a 3 × 3
    Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection padding was used to reduce artifacts. Rk
    denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer.
    uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2 .

    The network with 6 residual blocks consists of: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64,
    c7s1-3.

    The network with 9 residual blocks consists of: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256,
    R256, u128, u64, c7s1-3'

    Discriminator:

    'For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU
    layer with k filters and stride 2. After the last layer, we apply a convolution to produce a 1-dimensional output. We
    do not use InstanceNorm for the first C64 layer. We use leaky ReLUs with a slope of 0.2.
    The discriminator architecture is: C64-C128-C256-C512'

    """

    def _residual_block(res_x):
        """Residual block - 256 filters - 3x3 kernel size
        Following the guidelines from http://torch.ch/blog/2016/02/04/resnets.html
        """

        h = res_x

        h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')(h)
        h = tfa.layers.InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')(h)
        h = tfa.layers.InstanceNormalization()(h)

        return tf.keras.layers.add([res_x, h])

    # input
    x = inputs = tf.keras.Input(shape=[256, 256, 3])

    # c7s1-64 - 7x7 Convolution-InstanceNorm-ReLU (stride=1)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # d128 - 3x3 Convolution-InstanceNorm-ReLU (kernel_size=3, strides=2)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # d256 - 3x3 Convolution-InstanceNorm-ReLU (kernel_size=3, strides=2)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 6 x R256 - Residual block with 3x3 conv
    for _ in range(6):
        x = _residual_block(x)

    # u128 - 3x3 fractional-strided-Convolution-InstanceNorm-ReLU
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # u64 - 3x3 fractional-strided-Convolution-InstanceNorm-ReLU
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # c7s1-3 7x7 Convolution-InstanceNorm-ReLU (stride=1)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.tanh(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

