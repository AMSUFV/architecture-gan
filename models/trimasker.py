import tensorflow as tf
from models.pix2pix import Pix2Pix
from utils import dataset_tool


class MaskReconstructor(Pix2Pix):
    def __init__(self):
        super().__init__()
        self.masker = tf.keras.models.load_model('../trained_models/masker_019')

    def _step(self, train_x, train_y, training=True):
        with tf.GradientTape(persistent=True) as tape:
            x_m = self.masker(train_x, training=False)
            y_m = self.masker(train_y, training=False)

            g_x_m = self.generator(x_m, training=training)

            d_y_m = self.discriminator([x_m, y_m], training=training)
            d_g_x_m = self.discriminator([x_m, g_x_m], training=training)

            loss_d = self.discriminator_loss(d_y_m, d_g_x_m)
            loss_g = self.generator_loss(d_g_x_m, g_x_m, y_m)

        if training:
            gradients_g = tape.gradient(loss_g, self.generator.trainable_variables)
            gradients_d = tape.gradient(loss_d, self.discriminator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_g, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_d, self.discriminator.trainable_variables))

            if self.log_dir is not None:
                self.train_disc_loss(loss_d)
                self.train_gen_loss(loss_g)
                self.train_real_acc(tf.ones_like(d_y_m), d_y_m)
                self.train_gen_acc(tf.zeros_like(d_g_x_m), d_g_x_m)

        if not training and self.log_dir is not None:
            self.val_disc_loss(loss_d)
            self.val_gen_loss(loss_g)
            self.val_real_acc(tf.ones_like(d_y_m), d_y_m)
            self.val_gen_acc(tf.zeros_like(d_g_x_m), d_g_x_m)

    def fit(self, train, validation=None, epochs=5):
        for epoch in range(epochs):
            for train_x, train_y in train:
                self._step(train_x, train_y, training=True)
            for val_x, val_y in validation:
                self._step(val_x, val_y, training=False)
            