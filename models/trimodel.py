import tensorflow as tf
from models.pix2pix import Pix2Pix
from utils import dataset_tool


class Trimodel:
    def __init__(self):
        model_source = Pix2Pix()

        self.g_segmenter = tf.keras.models.load_model('../trained_models/segmenter.h5')

        # from segmented ruins to segmented reconstructions
        self.g_seg = model_source.build_generator()
        self.d_seg = model_source.build_discriminator()

        # colored reconstruction
        self.g_rec = model_source.build_generator(heads=2)
        self.d_rec = model_source.build_discriminator()

        # loss
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # optimizers
        self.g_seg_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_seg_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.g_rec_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_rec_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def loss_discriminator(self, d_real_output, d_g_output):
        d_loss_real = self.bce(tf.ones_like(d_real_output), d_real_output)
        d_loss_gen = self.bce(tf.zeros_like(d_g_output), d_g_output)
        d_loss_total = 0.5 * (d_loss_real + d_loss_gen)
        return d_loss_total

    def loss_generator(self, d_g_output, g_output, y, l1_multiplier=100):
        g_loss = self.bce(tf.ones_like(d_g_output), d_g_output)
        g_l1_loss = tf.reduce_mean(tf.abs(y - g_output))
        g_loss_total = g_loss + l1_multiplier * g_l1_loss
        return g_loss_total

    @tf.function
    def _step(self, ruin, temple_color, temple, training=True):
        with tf.GradientTape(persistent=True) as tape:
            # segmentation
            ruin_segmented = self.g_segmenter(ruin, training=False)

            # segmented reconstruction
            ruin_segmented_reconstructed = self.g_seg(ruin_segmented, training=training)
            d_color_real = self.d_seg([ruin_segmented, temple_color], training=training)
            d_color_gen = self.d_seg([ruin_segmented, ruin_segmented_reconstructed], training=training)

            # real reconstruction
            reconstruction = self.g_rec([ruin, ruin_segmented_reconstructed], training=training)
            d_reconstructor_real = self.d_rec([ruin, temple], training=training)
            d_reconstructor_gen = self.d_rec([ruin, reconstruction], training=training)

            # losses
            loss_d_color_rec = self.loss_discriminator(d_color_real, d_color_gen)
            loss_d_real_rec = self.loss_discriminator(d_reconstructor_real, d_reconstructor_gen)

            loss_g_seg_rec = self.loss_generator(d_color_gen, ruin_segmented_reconstructed, temple_color, 200)
            loss_g_real_rec = self.loss_generator(d_reconstructor_gen, reconstruction, temple)

            loss_color_rec = 0.5 * (loss_g_real_rec + loss_g_seg_rec)

        if training:
            gradients_g_seg = tape.gradient(loss_color_rec, self.g_seg.trainable_variables)
            gradients_g_rec = tape.gradient(loss_g_real_rec, self.g_rec.trainable_variables)
            gradients_d_seg = tape.gradient(loss_d_color_rec, self.d_seg.trainable_variables)
            gradients_d_rec = tape.gradient(loss_d_real_rec, self.d_rec.trainable_variables)

            self.g_seg_optimizer.apply_gradients(zip(gradients_g_seg, self.g_seg.trainable_variables))
            self.d_seg_optimizer.apply_gradients(zip(gradients_d_seg, self.d_seg.trainable_variables))
            self.g_rec_optimizer.apply_gradients(zip(gradients_g_rec, self.g_rec.trainable_variables))
            self.d_rec_optimizer.apply_gradients(zip(gradients_d_rec, self.d_rec.trainable_variables))

    def fit(self, train_ds, val_ds, epochs=50, name=None):
        if name is not None:
            writer_train = tf.summary.create_file_writer(f'../logs/{name}/train')
            writer_val = tf.summary.create_file_writer(f'../logs/{name}/val')
        else:
            writer_train = None
            writer_val = None

        for epoch in range(epochs):
            for ruin, color, temple in train_ds:
                self._step(ruin, color, temple)
            for ruin, color, temple in val_ds:
                self._step(ruin, color, temple, training=False)

            self._log_images(train_ds, writer_train, epoch)
            self._log_images(val_ds, writer_val, epoch)

    def _log_images(self, dataset, writer, epoch):
        for ruin, _, _ in dataset.take(1):
            segmentation = self.g_segmenter(ruin, training=False)
            segmented_reconstruction = self.g_seg(segmentation, training=False)
            reconstruction = self.g_rec([ruin, segmented_reconstruction], training=False)

            stack = tf.stack([segmentation, segmented_reconstruction, reconstruction], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('pipeline', stack, step=epoch, max_outputs=3)


if __name__ == '__main__':
    trimodel = Trimodel()
    temples = ['temple_1', 'temple_5', 'temple_6', 'temple_9']
    dataset_tool.setup_paths('../dataset')
    train, validation = dataset_tool.get_dataset_dual_input(temples, split=0.3, repeat=2)
    trimodel.fit(train, validation, name='trimodel_1569')
