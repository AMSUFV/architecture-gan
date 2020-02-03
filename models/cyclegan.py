from models.pix2pix import CustomPix2Pix
import tensorflow as tf


# TODO: revisar el paper de cycleGAN y entender los bloques residuales
class CycleGAN(CustomPix2Pix):
    def __init__(self, log_dir):
        super(CycleGAN, self).__init__(log_dir=log_dir)
        self.generator = ''
        self.discriminator = ''

        self.LAMBDA = 10

        # Generators
        # Given ruins or temples, the segmenter segments them into colors
        self.segmenter = self.build_generator()
        # Given ruins or temple segmentations, the desegmenter turns them back into their actual colors
        self.desegmenter = self.build_generator()
        # Given ruins or temple segmentations, the reconstructor turns them into temple reconstruction segmentations
        # full temples should remain untouched
        # self.reconstructor = self.build_generator()

        # Discriminators
        self.segmenter_disc = self.build_discriminator(target=False)
        self.desegmenter_disc = self.build_discriminator(target=False)
        # self.reconstructor = self.build_discriminator()

        # optimizers
        self.segmenter_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.desegmenter_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.segmenter_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.desegmenter_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def generator_loss(self, disc_output):
        return self.loss_object(tf.ones_like(disc_output), disc_output)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_loss = real_loss + generated_loss
        return total_loss * 0.5

    def cycle_loss(self, real, cycled):
        loss = tf.reduce_mean(tf.abs(real - cycled))
        return self.LAMBDA * loss

    def identity_loss(self, real, same):
        loss = tf.reduce_mean(tf.abs(real - same))
        return self.LAMBDA * 0.5 * loss

    @tf.function
    def train_step(self, train_x, train_y):
        # train_x: ruins and temples
        # train_y: their segmentations
        # train_z: complete temples
        with tf.GradientTape(persistent=True) as tape:
            # segmented ruins/temples
            x_segmented = self.segmenter(train_x, training=True)
            x_cycled = self.desegmenter(x_segmented, training=True)

            # desegmented segmentations
            y_desegmented = self.desegmenter(train_y, training=True)
            y_cycled = self.segmenter(y_desegmented, training=True)

            # reconstructions; the reconstructor should learn how to reconstruct from "ideal" images and the
            # generated segmentations
            # x_segmented_reconstruction = self.reconstructor(x_segmented, training=True)
            # y_reconstruction = self.reconstructor(train_y, training=True)

            # identity loss; segmenter and desegmenter shouldn't change images of the target domain
            y_identity = self.segmenter(train_y, training=True)
            x_identity = self.desegmenter(train_x, training=True)

            # discriminators
            disc_segmentation_real = self.segmenter_disc(train_y, training=True)
            disc_segmentation_gen = self.segmenter_disc(x_segmented, training=True)

            disc_desegmentation_real = self.desegmenter_disc(train_x, training=True)
            disc_desegmentation_gen = self.desegmenter_disc(y_desegmented, training=True)

            # disc_reconstruction_real = self.reconstructor(train_z, training=True)
            # disc_reconstruction_gen = self.reconstructor(x_segmented_reconstruction, training=True)

            segmenter_loss = self.generator_loss(disc_segmentation_gen)
            desegmenter_loss = self.generator_loss(disc_desegmentation_gen)

            # cycle loss
            total_cycle_loss = self.cycle_loss(train_x, x_cycled) + self.cycle_loss(train_y, y_cycled)

            # total gen loss
            total_segmenter_loss = segmenter_loss + total_cycle_loss + self.identity_loss(train_y, y_identity)
            total_desegmenter_loss = desegmenter_loss + total_cycle_loss + self.identity_loss(train_x, x_identity)

            segmenter_disc_loss = self.discriminator_loss(disc_segmentation_real, disc_segmentation_gen)
            desegmenter_disc_loss = self.discriminator_loss(disc_desegmentation_real, disc_desegmentation_gen)

        # gradients
        segmenter_gradients = tape.gradient(total_segmenter_loss, self.segmenter.trainable_variables)
        desegmenter_gradients = tape.gradient(total_desegmenter_loss, self.desegmenter.trainable_variables)
        segmenter_disc_gradients = tape.gradient(segmenter_disc_loss, self.segmenter_disc.trainable_variables)
        desegmenter_disc_gradients = tape.gradient(desegmenter_disc_loss, self.desegmenter_disc.trainable_variables)

        self.segmenter_optimizer.apply_gradients(zip(segmenter_gradients, self.segmenter.trainable_variables))
        self.desegmenter_optimizer.apply_gradients(zip(desegmenter_gradients, self.desegmenter.trainable_variables))
        self.segmenter_disc_optimizer.apply_gradients(zip(segmenter_disc_gradients, self.segmenter_disc.trainable_variables))
        self.desegmenter_disc_optimizer.apply_gradients(zip(desegmenter_disc_gradients, self.desegmenter_disc.trainable_variables))

    def fit(self, train_ds, test_ds, epochs, save_path=None):
        for epoch in range(epochs):
            for input_image, target_image in train_ds:
                self.train_step(input_image, target_image)

            with self.train_summary_writer.as_default():
                stack = self.get_tb_stack(train_ds)
                tf.summary.image('train', stack, max_outputs=4, step=epoch)

            with self.val_summary_writer.as_default():
                stack = self.get_tb_stack(test_ds)
                tf.summary.image('validation', stack, max_outputs=4, step=epoch)

    def get_tb_stack(self, dataset):
        for x, y in dataset.take(1):
            # x is a ruin/temple
            # y is a segmentation
            segmented = self.segmenter(x, training=False)
            desegmented = self.desegmenter(y, training=False)
            stack = tf.stack([x, segmented, desegmented, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            return stack


if __name__ == '__main__':
    cyclegan = CycleGAN(log_dir=r'logs\\cyclegan')
    train, test = cyclegan.get_complete_datset(temples=['temple_0'], mode='ruins_to_temples')
    cyclegan.fit(train, test, 50)
