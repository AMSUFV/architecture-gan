from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import datetime
import tensorflow as tf

from utils import pix2pix_preprocessing as preprocessing


# Convenience functions
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


# TODO add training checkpoints
class Pix2Pix:
    """Pix2Pix tailored to the needs of the ARQGAN project.
    """
    def __init__(self, *, gen_path: str = None, disc_path: str = None, log_dir: str = None):
        """Initialization of the Pix2Pix object. If paths for the generator and the discriminator are not specified,
        new ones will be created. Keeping logs of the training is recomended but optional.

        :param gen_path: str. Path to the generator (.h5) model. If not provided, a new one will be created.
        :param disc_path: str. Path to the discriminator (.h5) model. If not provided, a new one will be created.
        :param log_dir: str. Path to the log folder, optional. If specified, Tensorboard logs will be created in the
        target folder.
        """

        self.generator = self._set_weights(gen_path, self.build_generator)
        self.discriminator = self._set_weights(disc_path, self.build_discriminator)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.LAMBDA = 100

        # TODO: Think of a better way to handle this to keep single responsibility; e.g. create a diferent class.
        # Tensorboard
        self.log_dir = log_dir
        if self.log_dir is not None:
            # Writers
            current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            self.train_summary_writer = self._get_summary_writer(self.log_dir, current_time, 'train')
            self.val_summary_writer = self._get_summary_writer(self.log_dir, current_time, 'validation')

            # Metrics
            self.train_disc_loss = tf.keras.metrics.Mean('loss disc', dtype=tf.float32)
            self.train_gen_loss = tf.keras.metrics.Mean('loss gen', dtype=tf.float32)
            self.train_real_acc = tf.keras.metrics.BinaryAccuracy('accuracy real')
            self.train_gen_acc = tf.keras.metrics.BinaryAccuracy('accuracy generated')

            self.val_disc_loss = tf.keras.metrics.Mean('loss disc', dtype=tf.float32)
            self.val_gen_loss = tf.keras.metrics.Mean('loss gen', dtype=tf.float32)
            self.val_real_acc = tf.keras.metrics.BinaryAccuracy('accuracy real')
            self.val_gen_acc = tf.keras.metrics.BinaryAccuracy('accuracy generated')

            self.train_metrics = [self.train_disc_loss,
                                  self.train_gen_loss,
                                  self.train_real_acc,
                                  self.train_gen_acc]

            self.val_metrics = [self.val_disc_loss,
                                self.val_gen_loss,
                                self.val_real_acc,
                                self.val_gen_acc]

    @staticmethod
    def _get_summary_writer(path: str, time: str = None, name: str = 'train'):
        """Summary writer creation method. Creates and returns a Tensorboard summary writer for the specified path.

        :param path: str. Path to the target directory. Tensorboard logs will be written there.
        :param time: str. Time of creation for the logs. A way to differentiate between different runs.
        :param name: str. Name for the writer. Recommended names are train and validation.
        :return: Tensorboard summary writer
        """
        if time is None:
            time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = f'{path}\\{time}\\{name}'
        writer = tf.summary.create_file_writer(log_path)
        return writer

    @staticmethod
    def _set_weights(path: str = None, func=None):
        """Model creation method, not ment to be used outside of __init__. Returns a model either from a path or from
        a model creation function. If a path is not specified, said function must be.

        :param path: str. Path to the (.h5) model.
        :param func: function. Model creation function.
        :return: Tensorflow Keras Model
        """
        if path is not None:
            return tf.keras.models.load_model(path)
        else:
            if func is None:
                raise Exception('If no model path is specified, a model creation function must be provided.')
            else:
                return func()

    @staticmethod
    def build_generator(input_shape: list = None):
        """Generator creation function. Creates a u-net pix2pix-like generator.

        :param input_shape: list.
        :return: Tensorflow Keras Model
        """
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
            upsample(64, 4)
        ]

        if input_shape is None:
            input_shape = [512, 256, 3]

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs

        # downsampling
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        concat = tf.keras.layers.Concatenate()

        skips = reversed(skips[:-1])
        # upsampling and connecting
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                               kernel_initializer=initializer, activation='tanh')
        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    @staticmethod
    def build_discriminator(target=False, initial_units=64, layers=4, output_layer=None):
        """Patch-like discriminator creation function.

        :param target:
        :param initial_units:
        :param layers:
        :param output_layer:
        :return:
        """
        return_target = False
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')

        if not target:
            x = inp
        else:
            target = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
            x = tf.keras.layers.concatenate([inp, target])
            return_target = True

        multipliyer = 1
        for layer in range(layers):
            if layer == 1:
                x = downsample(initial_units * multipliyer, 4, apply_batchnorm=False)(x)
                multipliyer *= 2
            else:
                x = downsample(initial_units * multipliyer, 4)(x)
                if multipliyer < 8:
                    multipliyer *= 2

        if not output_layer:
            last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)
        else:
            last = output_layer

        if return_target:
            return tf.keras.Model(inputs=[inp, target], outputs=last)
        else:
            return tf.keras.Model(inputs=inp, outputs=last)

    # Metrics
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, disc_generated_output, gen_output, target):
        g_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = g_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss

    @staticmethod
    def get_dataset(input_path, output_path, split=0.2, batch_size=1, file_shuffle=False, input_shape=None):
        if input_shape is None:
            input_shape = [512, 384]

        preprocessing.IMG_WIDTH, preprocessing.IMG_HEIGHT = input_shape

        buffer_size = len(input_path)

        # This (and pix2pix_preprocessing) assume .png images will be used
        input_path = glob.glob(input_path + r'\*.png')
        output_path = glob.glob(output_path + r'\*.png')

        input_dataset = tf.data.Dataset.list_files(input_path, shuffle=file_shuffle)
        output_dataset = tf.data.Dataset.list_files(output_path, shuffle=file_shuffle)
        combined_dataset = tf.data.Dataset.zip((input_dataset, output_dataset)).shuffle(buffer_size)

        # train/validation split
        # Assuming both lists (input and output path) are the same length
        validation_size = round(buffer_size * split)
        train_size = buffer_size - validation_size

        train_dataset = combined_dataset.take(train_size)
        validation_dataset = combined_dataset.skip(train_size)

        train_dataset = train_dataset.map(preprocessing.load_images_train).shuffle(train_size).batch(batch_size)
        validation_dataset = validation_dataset.map(preprocessing.load_images_test).shuffle(validation_size)\
            .batch(batch_size)

        return train_dataset, validation_dataset

    # Training functions
    @tf.function
    def train_step(self, train_x, train_y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(train_x, training=True)

            disc_real_output = self.discriminator([train_x, train_y], training=True)
            disc_generated_output = self.discriminator([train_x, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, train_y)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        if self.log_dir is not None:
            self.train_disc_loss(disc_loss)
            self.train_gen_loss(gen_loss)
            self.train_real_acc(tf.ones_like(disc_real_output), disc_real_output)
            self.train_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)

    def fit(self, train_ds, test_ds=None, epochs=5):
        for epoch in range(epochs):
            # Train
            for input_image, target in train_ds:
                self.train_step(input_image, target)
            # Validation
            if test_ds is not None:
                for input_image, target_image in test_ds:
                    self.validate(input_image, target_image)

            # Tensorboard
            if self.log_dir is not None:
                self._write_metrics(self.train_summary_writer, self.train_metrics, epoch)
                self._reset_metrics(self.train_metrics)
                self._train_predict(train_ds, self.train_summary_writer, epoch, 'train')

                if test_ds is not None:
                    self._write_metrics(self.val_summary_writer, self.val_metrics, epoch)
                    self._reset_metrics(self.val_metrics)
                    self._train_predict(test_ds, self.val_summary_writer, epoch, 'validation')

    def validate(self, test_in, test_out):
        gen_output = self.generator(test_in, training=False)

        disc_real_output = self.discriminator([test_in, test_out], training=False)
        disc_generated_output = self.discriminator([test_in, gen_output], training=False)

        gen_loss = self.generator_loss(disc_generated_output, gen_output, test_out)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        if self.log_dir is not None:
            self.val_disc_loss(disc_loss)
            self.val_gen_loss(gen_loss)
            self.val_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)
            self.val_real_acc(tf.ones_like(disc_real_output), disc_real_output)

    def predict(self, dataset, log_path, samples):
        writer = self._get_summary_writer(log_path, 'predict')
        if samples == 'all':
            target = dataset
        else:
            target = dataset.take(samples)

        for x, y in target.take(samples):
            prediction = self.generator(x, training=False)

            stack = tf.stack([x, prediction, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('predictions', stack)

    def _train_predict(self, dataset, writer, step, name='train'):
        for x, y in dataset.take(1):
            generated = self.generator(x, training=False)
            stack = tf.stack([x, generated, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image(name, stack, step=step, max_outputs=3)

    @staticmethod
    def _write_metrics(writer, metrics, epoch):
        with writer.as_default():
            for metric in metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)

    @staticmethod
    def _reset_metrics(metrics):
        for metric in metrics:
            metric.reset_states()


if __name__ == '__main__':
    in_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\temples_ruins\temple_0_ruins_0'
    out_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\temples\temple_0'
    mypix2pix = Pix2Pix(log_dir=r'..\logs\test')
    # TODO: Maybe not initialize gen and disc in __init__
    mypix2pix.generator = mypix2pix.build_generator(input_shape=[1024, 768, 3])
    train, val = mypix2pix.get_dataset(in_path, out_path, input_shape=[1024, 768])
    mypix2pix.fit(train, val, epochs=50)
