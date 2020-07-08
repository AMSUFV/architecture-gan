import tensorflow as tf
from tensorflow import keras

from keras_models.pix2pix import Pix2Pix
from keras_parts.callbacks import ImageSampling
from keras_parts.losses import Pix2PixLosses
from keras_parts import pix2pix_generator, pix2pix_discriminator
from utils import data


generator = pix2pix_generator(input_shape=(256, 256, 3))
discriminator = pix2pix_discriminator(input_shape=(256, 256, 3))

g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

pix2pix = Pix2Pix(generator, discriminator)
pix2pix.compile(
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    g_loss_fn=Pix2PixLosses.loss_g,
    d_loss_fn=Pix2PixLosses.loss_d,
)


x = tf.random.normal((10, 256, 256, 3))
y = tf.ones_like(x, dtype=tf.float32)
x = tf.data.Dataset.from_tensor_slices(x).batch(1)
y = tf.data.Dataset.from_tensor_slices(y).batch(1)
dataset = tf.data.Dataset.zip((x, y))

# callbacks
# tensorboard = keras.callbacks.TensorBoard()
# image_sampling = ImageSampling(images=dataset.take(5), log_dir='logs')

pix2pix.fit(
    dataset,
    epochs=1,
    validation_data=dataset,
)
# pix2pix.evaluate(dataset)
