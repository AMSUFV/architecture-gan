import tensorflow as tf
from deprecated.models.pix2pix import Pix2Pix
from deprecated.utils import dataset_tool, custom_preprocessing as cp


class HybridReconstructor(Pix2Pix):
    @tf.function
    def train_step(self, ruin, color, temple):
        with tf.GradientTape(persistent=True) as tape:
            gen_output = self.generator([ruin, color], training=True)

            disc_real = self.discriminator([ruin, temple], training=True)
            disc_generated = self.discriminator([ruin, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated, gen_output, temple)
            disc_loss = self.discriminator_loss(disc_real, disc_generated)

        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        if self.log_dir is not None:
            self.train_disc_loss(disc_loss)
            self.train_gen_loss(gen_loss)
            self.train_real_acc(tf.ones_like(disc_real), disc_real)
            self.train_gen_acc(tf.zeros_like(disc_generated), disc_generated)

    def fit(self, train_ds, test_ds=None, epochs=100):
        for epoch in range(epochs):
            # Train
            for ruin, color, temple in train_ds:
                self.train_step(ruin, color, temple)

            if test_ds is not None:
                self.validate(test_ds)

            self._metric_update(train_ds, test_ds, epoch)
            if self.log_dir is not None:
                self._train_predict(train_ds, self.train_summary_writer, epoch, 'train')
                if test_ds is not None:
                    self._train_predict(test_ds, self.val_summary_writer, epoch, 'validation')

    def validate(self, test):
        for test_ruin, test_color, test_temple in test:
            gen_output = self.generator([test_ruin, test_color], training=False)

            disc_real_output = self.discriminator([test_ruin, test_temple], training=False)
            disc_generated_output = self.discriminator([test_ruin, gen_output], training=False)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, test_temple)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            if self.log_dir is not None:
                self.val_disc_loss(disc_loss)
                self.val_gen_loss(gen_loss)
                self.val_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)
                self.val_real_acc(tf.ones_like(disc_real_output), disc_real_output)

    def _train_predict(self, dataset, writer, step, name='train'):
        for ruin, color, temple in dataset.take(1):
            generated = self.generator([ruin, color], training=False)
            stack = tf.stack([ruin, color, generated, temple], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image(name, stack, step=step, max_outputs=4)


def main(training_name, temples):
    # TODO: standarise r and f
    log_path = f'../logs/{training_name}'
    cp.RESIZE_FACTOR = 1.3

    train, val = dataset_tool.get_dataset_dual_input(temples=temples, split=0.3, repeat=2)

    reconstructor = HybridReconstructor(log_dir=log_path, autobuild=False)

    reconstructor.build_generator(heads=2, inplace=True)
    reconstructor.build_discriminator()

    reconstructor.fit(train, val, epochs=10)


if __name__ == '__main__':
    main(training_name='test', temples=['temple_0'])
