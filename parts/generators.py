import tensorflow as tf
import tensorflow_addons as tfa

from parts.blocks import downsample, upsample


# TODO: add reflection padding
def resnet(input_shape=(512, 512, 3), dim=64, downsamplings=2, res_blocks=9):
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
        filters = res_x.shape[-1]
        h = res_x

        h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(h)
        h = tfa.layers.InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(h)
        h = tfa.layers.InstanceNormalization()(h)

        return tf.keras.layers.add([res_x, h])

    # input
    x = inputs = tf.keras.Input(shape=input_shape)

    # c7s1-64 - 7x7 Convolution-InstanceNorm-ReLU (stride=1)
    x = tf.keras.layers.Conv2D(filters=dim, kernel_size=7, strides=1, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # d128 - 3x3 Convolution-InstanceNorm-ReLU (kernel_size=3, strides=2)
    # d256 - 3x3 Convolution-InstanceNorm-ReLU (kernel_size=3, strides=2)
    for _ in range(downsamplings):
        dim *= 2
        x = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, strides=2, padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # 6-9 x R256 - Residual block with 3x3 conv
    for _ in range(res_blocks):
        x = _residual_block(x)

    # u128 - 3x3 fractional-strided-Convolution-InstanceNorm-ReLU
    # u64 - 3x3 fractional-strided-Convolution-InstanceNorm-ReLU
    for _ in range(downsamplings):
        dim //= 2
        x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # c7s1-3 7x7 Convolution-InstanceNorm-ReLU (stride=1)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.tanh(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def pix2pix(input_shape=None, heads=1, dim=64, down_blocks=8, downsamplings=4, norm_type='batchnorm', activation='tanh'):

    if input_shape is None:
        input_shape = (None, None, 3)

    down_stack = [
        dict(filters=64, kernel_size=4, apply_norm=False),
        dict(filters=128, kernel_size=4, apply_norm=True, norm_type=norm_type),
        dict(filters=256, kernel_size=4, apply_norm=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_norm=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_norm=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_norm=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_norm=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_norm=True, norm_type=norm_type),
    ]

    up_stack = [
        dict(filters=512, kernel_size=4, apply_dropout=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_dropout=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_dropout=True, norm_type=norm_type),
        dict(filters=512, kernel_size=4, apply_dropout=False, norm_type=norm_type),
        dict(filters=256, kernel_size=4, apply_dropout=False, norm_type=norm_type),
        dict(filters=128, kernel_size=4, apply_dropout=False, norm_type=norm_type),
        dict(filters=64, kernel_size=4, apply_dropout=False, norm_type=norm_type),
    ]

    input_layers = []
    for n in range(heads):
        input_layers.append(tf.keras.layers.Input(shape=input_shape))

    if heads > 1:
        x = tf.keras.layers.concatenate(input_layers)
    else:
        x = input_layers[0]

    # downsampling
    skips = []
    for block in down_stack:
        x = downsample(x,
                       block['filters'],
                       block['kernel_size'],
                       apply_norm=block['apply_norm'])
        skips.append(x)

    skips = reversed(skips[:-1])

    # upsampling and connecting
    for up, skip in zip(up_stack, skips):
        x = upsample(x, up['filters'], up['kernel_size'], apply_dropout=up['apply_dropout'])
        x = tf.keras.layers.Concatenate()([x, skip])

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(filters=3,
                                           kernel_size=4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation=activation)
    x = last(x)

    return tf.keras.Model(inputs=input_layers, outputs=x)
