from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import tensorflow as tf
from utils import dataset_tool


def downsample(filters: int, size: int, apply_batchnorm=True):
    """Convenience function for the creation of a downsampling block made of a 2D Convolutional layer, an optional
    Batch Normalization layer and a Leaky ReLU activation function.

    :param filters: int. Determines the number of filters in the Conv2D layer.
    :param size: int. Determines the size of said filters.
    :param apply_batchnorm: Wether or not to apply a BatchNormalization layer to the activations.
    :return: A sequential model consisting of a Conv2D, an optional BatchNormalization and a LeakyReLU
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters: int, size: int, apply_dropout=False):
    """Convenience function for the creation of an umpsampling block made of a Transposed 2D Convolutional layer,
    a Batch Normalization layer, an optional Droput layer and a Leaky ReLU activation function.

    :param filters: int. Determines the number of filters in the Conv2DTranspose layer.
    :param size: int. Determines the size of said filters.
    :param apply_dropout: Wether or not to apply a Dropout to the activations.
    :return: A sequential model consisting of a Conv2D, an optional BatchNormalization and a LeakyReLU
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def generator(input_shape: list = None, heads=1, out_dims=3):
    if input_shape is None:
        input_shape = [None, None, 3]

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
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
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # upsampling and connecting
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(out_dims, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')
    x = last(x)

    return tf.keras.Model(inputs=input_layers, outputs=x)


def discriminator(input_shape=None, initial_units=64, layers=4):
    if input_shape is None:
        input_shape = [None, None, 3]

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    target_image = tf.keras.layers.Input(shape=input_shape, name='target_image')
    x = tf.keras.layers.concatenate([inp, target_image])

    multipliyer = 1
    for layer in range(layers):
        if layer == 1:
            x = downsample(initial_units * multipliyer, 4, apply_batchnorm=False)(x)
            multipliyer *= 2
        else:
            x = downsample(initial_units * multipliyer, 4)(x)
            if multipliyer < 8:
                multipliyer *= 2

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, activation='sigmoid')(x)

    return tf.keras.Model(inputs=[inp, target_image], outputs=last)

# ----------------------------------------------------------------------------------------------------------------------


class Pix2Pix:
    def __init__(self, g_params=None, d_params=None):
        if g_params is None:
            g_params = dict(input_shape=[None, None, 3], out_dims=3, heads=1)
        if d_params is None:
            d_params = dict(input_shape=[None, None, 3])

        self.generator = generator(g_params['input_shape'], g_params['heads'], g_params['out_dims'])
        self.discriminator = discriminator(input_shape=d_params['input_shape'])

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer_g = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.optimizer_d = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def _loss_d(self, d_y, d_g_x):
        loss_y = self.loss_object(tf.ones_like(d_y), d_y)
        loss_g_x = self.loss_object(tf.zeros_like(d_g_x), d_g_x)
        loss_total = 0.5 * (loss_y + loss_g_x)
        return loss_total

    def _loss_g(self, d_g_x, **kwargs):
        """Base generator loss function.

        :param d_g_x: discriminator output for the generator's output.
        :param kwargs: for the base class, accepted kwargs are y (expected output) and g_x (generator output)
        :return: generator loss
        """
        loss_d_g_x = self.loss_object(tf.ones_like(d_g_x), d_g_x)
        loss_l1 = tf.reduce_mean(tf.abs(kwargs['y'] - kwargs['g_x']))
        loss_total = loss_d_g_x + kwargs['multiplier'] * loss_l1
        return loss_total, loss_l1

    @tf.function()
    def _step(self, train_x, train_y, training=True):
        with tf.GradientTape(persistent=True) as tape:
            g_x = self.generator(train_x, training=training)

            d_g_x = self.discriminator([train_x, g_x], training=training)
            d_y = self.discriminator([train_x, train_y], training=training)

            g_loss, l1_loss = self._loss_g(d_g_x, y=train_y, g_x=g_x, multiplier=100)
            d_loss = self._loss_d(d_y, d_g_x)

        if training:
            self._gradient_update(tape, g_loss, self.generator, self.optimizer_g)
            self._gradient_update(tape, d_loss, self.discriminator, self.optimizer_d)
            # gradients_g = tape.gradient(g_loss, self.generator.trainable_variables)
            # gradients_d = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # self.optimizer_g.apply_gradients(zip(gradients_g, self.generator.trainable_variables))
            # self.optimizer_d.apply_gradients(zip(gradients_d, self.discriminator.trainable_variables))

        metrics_names = ['loss_gen_total', 'loss_gen_l1', 'loss_disc']
        metrics = [g_loss, l1_loss, d_loss]
        return metrics_names, metrics

    @staticmethod
    def _gradient_update(tape, loss, model, optimizer):
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    @staticmethod
    def _write_metrics(writer, metrics_names, metrics, epoch):
        if writer is not None:
            with writer.as_default():
                for name, metric in zip(metrics_names, metrics):
                    tf.summary.scalar(name, metric.result(), epoch)

    def fit(self, train, validation, epochs=1, log_dir=None):
        writer_train, writer_val = None, None
        metrics_names, metrics = None, None
        if log_dir is not None:
            writer_train, writer_val = self._get_writers(log_dir)

        for epoch in range(epochs):
            for train_x, train_y in train:
                metrics_names, metrics = self._step(train_x, train_y, training=True)
            # self._write_metrics(writer_train, metrics_names, metrics, epoch)
            self._log_images(train, writer_train, epoch)

            for val_x, val_y in validation:
                self._step(val_x, val_y, training=False)
            # self._write_metrics(writer_val, metrics_names, metrics, epoch)
            self._log_images(validation, writer_val, epoch)

    def _log_images(self, dataset, writer, epoch):
        if writer is not None:
            x, _ = next(dataset.take(1).__iter__())
            g_x = self.generator(x, training=False)
            stack = tf.stack([x, g_x], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image('prediction', stack, step=epoch, max_outputs=2)

    @staticmethod
    def _get_writers(log_dir):
        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = f'{log_dir}/{time}/'
        writer_train = tf.summary.create_file_writer(log_path + 'train')
        writer_val = tf.summary.create_file_writer(log_path + 'validation')

        return writer_train, writer_val


if __name__ == '__main__':
    dataset_tool.setup_paths('../dataset')
    train_ds, val_ds = dataset_tool.get_dataset_segmentation(['temple_0'], repeat=2)
    pix2pix = Pix2Pix()
    pix2pix.fit(train_ds, val_ds, epochs=10, log_dir='../logs/test')
