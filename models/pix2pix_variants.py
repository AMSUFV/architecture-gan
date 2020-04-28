from datetime import datetime
import tensorflow as tf

from parts.discriminators import patch as discriminator
from parts.generators import pix2pix as generator
from parts import losses


class Pix2Pix:
    def __init__(self, input_shape=(None, None, 3), norm_type='batchnorm', heads=1):
        self.name = 'pix2pix'
        self.discriminator = discriminator(input_shape=input_shape, norm_type=norm_type)
        self.generator = generator(input_shape=input_shape, norm_type=norm_type, heads=heads)
        self.loss_d, self.loss_g = losses.pix2pix()
        self.g_optimizer = self.d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def __call__(self, image):
        return self.generator(image, training=False)

    def summary(self, to_file=False):
        self.generator.summary()
        self.discriminator.summary()
        if to_file:
            tf.keras.utils.plot_model(self.generator, to_file='generator.png')
            tf.keras.utils.plot_model(self.discriminator, to_file='discriminator.png')

    @staticmethod
    def write(metrics_dict, step=None, name='summary'):
        with tf.name_scope(name):
            for name, data in metrics_dict.items():
                tf.summary.scalar(name, data, step=step)

    @tf.function
    def train_g(self, x, y):
        with tf.GradientTape() as t:
            gx = self.generator(x, training=True)
            dgx = self.discriminator([x, gx], training=True)
            g_loss, l1_loss = self.loss_g(y, gx, dgx)

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return gx, dict(g_loss=g_loss,
                        l1_loss=l1_loss)

    @tf.function
    def train_d(self, x, gx, y):
        with tf.GradientTape() as t:
            dy = self.discriminator([x, y], training=True)
            dgx = self.discriminator([x, gx], training=True)
            y_loss, gx_loss = self.loss_d(dy, dgx)
            d_loss = y_loss + gx_loss

        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        return dict(y_loss=y_loss,
                    gx_loss=gx_loss,
                    d_loss=d_loss)

    @tf.function
    def train_step(self, x, y):
        gx, g_dict = self.train_g(x, y)
        d_dict = self.train_d(x, gx, y)

        return g_dict, d_dict

    def fit(self, dataset, epochs, path=None):
        time = datetime.now().strftime('%Y%m%d-%H%M%S')
        writer = tf.summary.create_file_writer(path + f'{self.name}/{time}/train')

        with writer.as_default():
            for _ in range(epochs):
                for x, y in dataset:
                    g_dict, d_dict = self.train_step(x, y)
                    self.write(g_dict, step=self.g_optimizer.iterations, name='g_losses')
                    self.write(d_dict, step=self.g_optimizer.iterations, name='d_losses')
