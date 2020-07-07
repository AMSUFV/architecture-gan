import tensorflow as tf
from tensorflow.keras import layers


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
