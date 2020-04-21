import tensorflow as tf
from tensorflow.keras import losses


def pix2pix():
    bce = losses.BinaryCrossentropy(from_logits=True)

    def loss_d(dy, dgx):
        loss_y = bce(tf.ones_like(dy), dy)
        loss_gx = bce(tf.zeros_like(dgx), dgx)
        return loss_y, loss_gx

    def loss_g(y, gx, dgx):
        loss_d_g_x = bce(tf.ones_like(dgx), dgx)
        loss_l1 = tf.reduce_mean(tf.abs(y - gx))
        loss_total = loss_d_g_x + 100 * loss_l1
        return loss_total, loss_l1

    return loss_d, loss_g
