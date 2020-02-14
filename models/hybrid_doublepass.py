from models.hybrid_reconstuctor import HybridReconstuctor
from utils import custom_preprocessing as cp

import tensorflow as tf


class HyDoublePass(HybridReconstuctor):
    @tf.function
    def train_step(self, ruin, temple, color):
        with tf.GradientTape(persistent=True) as tape:
            gen_output = self.generator([ruin, color], training=True)
            gen_second_output = self.generator([gen_output, color], training=True)

            disc_real = self.discriminator([ruin, temple], training=True)
            disc_generated = self.discriminator([ruin, gen_output], training=True)

            disc_second_generated = self.discriminator([ruin, gen_second_output], training=False)

            gen_loss = self.generator_loss(disc_generated, gen_output, temple)
            gen_second_loss = self.generator_loss(disc_second_generated, gen_second_output, temple, lmda=200)

            disc_loss = self.discriminator_loss(disc_real, disc_generated)

        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        gen_second_gradients = tape.gradient(gen_second_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gen_second_gradients), self.generator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        if self.log_dir is not None:
            self.train_disc_loss(disc_loss)
            self.train_gen_loss(gen_loss)
            self.train_real_acc(tf.ones_like(disc_real), disc_real)
            self.train_gen_acc(tf.zeros_like(disc_generated), disc_generated)


def main():
    log_path = r'..\logs\second_pass_125'
    ds_path = r'..\dataset'
    temple_list = ['temple_1', 'temple_2', 'temple_5']

    cp.RESIZE_FACTOR = 1.3
    reconstructor = HybridReconstuctor(log_dir=log_path, autobuild=True)
    train, validation = reconstructor.get_dataset(temples=temple_list, dataset_path=ds_path, split=0.25)
    reconstructor.fit(train, validation, 50)
    tf.keras.models.save_model(reconstructor.generator, '../trained_models/second_pass_125.h5')


if __name__ == '__main__':
    main()
