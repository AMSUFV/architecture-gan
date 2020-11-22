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

    @staticmethod
    def area_loss(y, gx, area):
        """Loss aimed at pinpointing the differences in the area to be reconstructed
        @param y: Tensor. Expected image.
        @param gx: Tensor. Predicted image.
        @param area: Tensor. Area where the reconstruction is happening, matrix of True/False or 1/0.
        @return: loss: Tensor. L1 distance between y_area and gx_area.
        """
        diff = tf.abs(y - gx)
        return tf.reduce_sum(
            tf.where(mask == 1, diff, 0) / tf.cast(tf.reduce_sum(area), tf.float32)
        )
