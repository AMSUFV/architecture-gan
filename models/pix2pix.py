from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import glob

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
    def __init__(self, *, gen_path: str = None, disc_path: str = None, log_dir: str = None, autobuild=False):
        """Initialization of the Pix2Pix object. If paths for the generator and the discriminator are not specified,
        new ones will be created. Keeping logs of the training is recomended but optional.

        :param gen_path: str. Path to the generator (.h5) model. If not provided, a new one will be created when fit is
        called.
        :param disc_path: str. Path to the discriminator (.h5) model. If not provided, a new one will be created when
        fit is called.
        :param log_dir: str. Path to the log folder, optional. If specified, Tensorboard logs will be created in the
        target folder.
        """
        self.generator = None
        self.discriminator = None

        self.generator = self._set_weights(gen_path, self.build_generator, autobuild)
        self.discriminator = self._set_weights(disc_path, self.build_discriminator, autobuild)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

        self.LAMBDA = 100

        # Checkpoints
        # checkpoint_dir = '../training_checkpoints'
        # checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        # checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
        #                                  discriminator_optimizer=self.discriminator_optimizer,
        #                                  generator=self.generator,
        #                                  discriminator=self.discriminator)

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
    def _set_weights(path: str = None, func=None, autobuild=False):
        """Model creation method, not ment to be used outside of __init__. Returns a model either from a path or from
        a model creation function. If a path is not specified, said function must be.

        :param path: str. Path to the (.h5) model.
        :param func: function. Model creation function.
        :param autobuild: boolean. Wether or not to build the predefined model
        :return: Tensorflow Keras Model
        """
        if path is not None:
            return tf.keras.models.load_model(path)
        elif autobuild:
            return func()
        else:
            return None

    def build_generator(self, input_shape: list = None, heads=1, inplace=False):
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
        last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                               kernel_initializer=initializer, activation='tanh')
        x = last(x)

        if inplace:
            self.generator = tf.keras.Model(inputs=input_layers, outputs=x)
        return tf.keras.Model(inputs=input_layers, outputs=x)

    def build_discriminator(self, input_shape=None, initial_units=64, layers=4, inplace=True):
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

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)

        if inplace:
            self.discriminator = tf.keras.Model(inputs=[inp, target_image], outputs=last)
        else:
            return tf.keras.Model(inputs=[inp, target_image], outputs=last)

    # Metrics
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

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
        validation_dataset = validation_dataset.map(preprocessing.load_images_test).shuffle(validation_size) \
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
                    self.validate(test_ds)

            self._metric_update(train_ds, test_ds, epoch)

            if epoch % 10 == 0:
                self._train_predict(train_ds, self.train_summary_writer, epoch, 'train')
                if test_ds is not None:
                    self._train_predict(test_ds, self.val_summary_writer, epoch, 'validation')

    def _metric_update(self, train_ds, test_ds, epoch):
        if self.log_dir is not None:
            self._write_metrics(self.train_summary_writer, self.train_metrics, epoch)
            self._reset_metrics(self.train_metrics)

            if test_ds is not None:
                self._write_metrics(self.val_summary_writer, self.val_metrics, epoch)
                self._reset_metrics(self.val_metrics)

    def validate(self, test):
        for test_in, test_out in test.take(1):
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
        """Prediction method ment to be used outside of training.

        :param dataset:
        :param log_path:
        :param samples:
        :return:
        """
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

    def _train_predict(self, dataset, writer, step: int, name='train'):
        """Method used during training to record the model's progress on image generation / translation tasks.

        :param dataset: tensorflow.Dataset. Dataset to take the samples from
        :param writer: tf.summary.writer. Writer to use when recording the progress.
        :param step: int. Used as step value when writing the metric.
        :param name: str. Name to be showed on Tensorboard.
        """
        for x, y in dataset.take(1):
            generated = self.generator(x, training=False)
            stack = tf.stack([x, generated, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image(name, stack, step=step, max_outputs=3)

    @staticmethod
    def _write_metrics(writer, metrics: list, step: int):
        """Convenience method for writing scalar metrics to Tensorboard

        :param writer: tf.summary.writer. Writer to use when updating the metrics.
        :param metrics: list. List tf.summary.scalar metrics to record.
        :param step: int. Used as step value when writing the metric.
        """
        with writer.as_default():
            for metric in metrics:
                tf.summary.scalar(metric.name, metric.result(), step=step)

    @staticmethod
    def _reset_metrics(metrics: list):
        """Convenience method for reseting Tensorboard metrics

        :param metrics: list. List of metrics to reset.
        """
        for metric in metrics:
            metric.reset_states()


if __name__ == '__main__':
    pix2pix = Pix2Pix(log_dir=r'..\logs\full_temple_train_pix2pix', autobuild=True)

