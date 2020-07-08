import tensorflow as tf
from tensorflow import keras


# noinspection PyAttributeOutsideInit,PyMethodOverriding
class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def call(self, inputs, training=False, disc_output=False):
        outputs = self.generator(inputs, training=training)
        if disc_output:
            dgx = self.discriminator([inputs, gx], training=training)
            outputs = (outputs, dgx)
        return outputs

    def test_step(self, data):
        x, y = data
        gx = self.generator(x, training=False)
        dy = self.discriminator([x, y], training=False)
        dgx = self.discriminator([x, gx], training=False)

        g_loss = self.g_loss_fn(y, gx, dgx)
        d_loss = self.d_loss_fn(dy, dgx)

        return {"g_loss": g_loss, "d_loss": d_loss}

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn):
        super(Pix2Pix, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

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

    def train_g(self, x, y):
        with tf.GradientTape() as t:
            gx = self.generator(x, training=True)
            dgx = self.discriminator([x, gx], training=True)
            g_loss = self.g_loss_fn(y, gx, dgx)

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )
        return gx, g_loss

    def train_step(self, images):
        x, y = images
        gx, g_loss = self.train_g(x, y)
        d_loss = self.train_d(x, gx, y)
        return {"g_loss": g_loss, "d_loss": d_loss}


class Assisted(Pix2Pix):
    def train_g(self, x, y):
        x1, x2 = x
        with tf.GradientTape() as t:
            gx = self.generator([x1, x2], training=True)
            dgx = self.discriminator([x1, gx], training=True)
            g_loss = self.g_loss_fn(y, gx, dgx)

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )
        return gx, g_loss

    def train_d(self, x, gx, y):
        x1, x2 = x
        with tf.GradientTape() as t:
            dy = self.discriminator([x1, y], training=True)
            dgx = self.discriminator([gx, y], training=True)
            d_loss = self.d_loss_fn(dy, dgx)
        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_variables)
        )
        return d_loss
