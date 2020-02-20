import tensorflow as tf
import datetime
from models.hybrid_reconstuctor import HybridReconstuctor
from models.pix2pix import downsample


class Classifier:
    def __init__(self, path_generator):
        self.generator = tf.keras.models.load_model(path_generator)

        self.discriminator = self.build_discriminator()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # tensorboard
        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer_train = tf.summary.create_file_writer(f'../logs/c2st/{time}/train')
        self.writer_val = tf.summary.create_file_writer(f'../logs/c2st/{time}/val')
        # metrics
        self.train_disc_loss = tf.keras.metrics.Mean('loss disc', dtype=tf.float32)
        self.train_acc_real = tf.keras.metrics.BinaryAccuracy('acc real', threshold=0.2)
        self.train_acc_gen = tf.keras.metrics.BinaryAccuracy('acc gen', threshold=0.2)

        self.val_disc_loss = tf.keras.metrics.Mean('loss disc', dtype=tf.float32)
        self.val_acc_real = tf.keras.metrics.BinaryAccuracy('acc real', threshold=0.2)
        self.val_acc_gen = tf.keras.metrics.BinaryAccuracy('acc gen', threshold=0.2)

        self.train_metrics = [self.train_disc_loss, self.train_acc_real, self.train_acc_gen]
        self.val_metrics = [self.val_disc_loss, self.val_acc_real, self.val_acc_gen]

    @staticmethod
    def build_discriminator(input_shape=None, initial_units=64, layers=4, inplace=True):
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

        return tf.keras.Model(inputs=[inp, target_image], outputs=last)

    def discriminator_loss(self, disc_real, disc_gen):
        loss_real = self.loss_object(tf.ones_like(disc_real), disc_real)
        loss_fake = self.loss_object(tf.zeros_like(disc_gen), disc_gen)
        loss_total = loss_real + loss_fake
        return loss_total

    @tf.function
    def _step(self, ruin, temple, color, training=True):
        with tf.GradientTape(persistent=True) as tape:
            gen_output = self.generator([ruin, color], training=False)

            disc_real = self.discriminator([ruin, temple], training=training)
            disc_gen = self.discriminator([ruin, gen_output], training=training)
            disc_loss = self.discriminator_loss(disc_real, disc_gen)

        if training:
            disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # tensorboard
        if training:
            self.train_disc_loss(disc_loss)
            self.train_acc_real(tf.ones_like(disc_real), disc_real)
            self.train_acc_gen(tf.zeros_like(disc_gen), disc_gen)
        else:
            self.val_disc_loss(disc_gen)
            self.val_acc_real(tf.ones_like(disc_real), disc_real)
            self.val_acc_gen(tf.zeros_like(disc_gen), disc_gen)

    def fit(self, dataset_train, dataset_val, epochs=50):
        for epoch in range(epochs):
            for ruin, temple, color in dataset_train:
                self._step(ruin, temple, color, training=True)

            for ruin, temple, color in dataset_val:
                self._step(ruin, temple, color, training=False)

            # tensorboard
            self._metric_update(self.writer_train, self.train_metrics, epoch)
            self._image_matrix(self.writer_train, dataset_train, epoch)
            self._metric_update(self.writer_val, self.val_metrics, epoch)
            self._image_matrix(self.writer_val, dataset_val, epoch)

    @staticmethod
    def _metric_update(writer, metrics, step):
        with writer.as_default():
            for metric in metrics:
                tf.summary.scalar(metric.name, metric.result(), step=step)
                metric.reset_states()

    def _image_matrix(self, writer, dataset, step):
        for ruin, temple, color in dataset.take(1):
            reconstruction = self.generator([ruin, color], training=False)

            markov_rf_real = self.discriminator([ruin, temple], training=False)
            markov_rf_fake = self.discriminator([ruin, reconstruction], training=False)

            markov_rf_real = tf.image.resize(markov_rf_real, [256, 512])
            markov_rf_fake = tf.image.resize(markov_rf_fake, [256, 512])

            stack_temple = tf.stack([reconstruction, temple], axis=0) * 0.5 + 0.5
            stack_temple = tf.squeeze(stack_temple)

            stack_mrf = tf.stack([markov_rf_fake, markov_rf_real], axis=0) * 0.5 + 0.5
            stack_mrf = tf.squeeze(stack_mrf, axis=1)

            with writer.as_default():
                tf.summary.image('models', stack_temple, step=step)
                tf.summary.image('markov random fields', stack_mrf, step=step)


if __name__ == '__main__':
    dataset_creator = HybridReconstuctor(autobuild=False)
    train, val = dataset_creator.get_dataset(temples=['temple_0'], ruins_per_temple=2)
    c2st = Classifier(path_generator='../trained_models/reconstructor.h5')
    c2st.fit(train, val, 50)
