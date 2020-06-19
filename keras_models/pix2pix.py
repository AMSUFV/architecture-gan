import tensorflow as tf
from tensorflow import keras


# noinspection PyAttributeOutsideInit,PyMethodOverriding
class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super(Pix2Pix, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def call(self, inputs, training=None, mask=None):
        image = self.generator(inputs, training=training)
        prediction = self.discriminator(image, training=training)
        return image, prediction

    def train_g(self, x, y):
        with tf.GradientTape() as t:
            gx = self.generator(x, training=True)
            dgx = self.discriminator([x, gx], training=True)
            g_loss, l1_loss = self.g_loss_fn(y, gx, dgx)

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )

        return gx, g_loss, l1_loss

    def train_d(self, x, gx, y):
        with tf.GradientTape() as t:
            dy = self.discriminator([x, y], training=True)
            dgx = self.discriminator([x, gx], training=True)
            d_loss = self.d_loss_fn(dy, dgx)

        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_variables)
        )

        return d_loss

    def train_step(self, images):
        x, y = images
        gx, g_loss, l1_loss = self.train_g(x, y)
        d_loss = self.train_d(x, gx, y)
        return {"d_loss": d_loss, "g_loss": g_loss, "l1_loss": l1_loss}
