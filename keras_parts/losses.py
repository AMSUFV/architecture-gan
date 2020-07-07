import tensorflow as tf
from tensorflow import keras

bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)


class Pix2PixLosses:
    @staticmethod
    def loss_g(y, gx, dgx):
        dgx_loss = bce_logits(tf.ones_like(dgx), dgx)
        l1_loss = tf.reduce_mean(tf.abs(y - gx))
        return dgx_loss + 100 * l1_loss

    @staticmethod
    def loss_d(dy, dgx):
        dy_loss = bce_logits(tf.ones_like(dy), dy)
        dgx_loss = bce_logits(tf.zeros_like(dgx), dgx)
        return (dy_loss + dgx_loss) / 2
