import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


# noinspection PyAttributeOutsideInit,PyMethodOverriding
class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def call(self, inputs, training=False, disc_output=False):
        outputs = self.generator(inputs, training=training)
        if disc_output:
            dgx = self.discriminator([inputs, outputs], training=training)
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

    def compile(self, g_optimizer=None, d_optimizer=None, g_loss_fn=None, d_loss_fn=None):
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
    def call(self, inputs, training=False, disc_output=False):
        outputs = self.generator(inputs, training=training)
        if disc_output:
            dgx = self.discriminator([inputs[0], outputs], training=training)
            outputs = (outputs, dgx)
        return outputs

    def train_step(self, images):
        *x, y = images
        gx, g_loss = self.train_g(x, y)
        d_loss = self.train_d(x[0], gx, y)
        return {"g_loss": g_loss, "d_loss": d_loss}

    def train_g(self, x, y):
        with tf.GradientTape() as t:
            gx = self.generator(x, training=True)
            dgx = self.discriminator([x[0], gx], training=True)
            g_loss = self.g_loss_fn(y, gx, dgx)
        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )
        return gx, g_loss

    def test_step(self, data):
        *x, y = data
        gx = self.generator(x, training=False)
        dy = self.discriminator([x[0], y], training=False)
        dgx = self.discriminator([x[0], gx], training=False)

        g_loss = self.g_loss_fn(y, gx, dgx)
        d_loss = self.d_loss_fn(dy, dgx)

        return {"g_loss": g_loss, "d_loss": d_loss}


class StepModel:
    """StepModel
    Model that takes ruin images as an input and is comprised of:
    - A segmenter that generates a segmentation of said ruins
    - A color reconstructor that reconstruct that segmentation
    - A reconstructor that takes both the ruins and the segmented reconstruction to generate the true-color
      reconstruction of the ruins
    This class is for testing purposes only.
    """

    def __init__(self, segmenter: str, color_reconstructor: str, reconstructor: str):
        self.segmenter = load_model(segmenter)
        self.color_reconstructor = load_model(color_reconstructor)
        self.reconstructor = load_model(reconstructor)

    def __call__(self, x, training=False):
        x_c = self.segmenter(x, training=training)
        x_rc = self.color_reconstructor(x_c, training=training)
        g_x = self.reconstructor([x, x_rc], training=training)
        return g_x
