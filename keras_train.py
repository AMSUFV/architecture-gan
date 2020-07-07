import tensorflow as tf
from tensorflow import keras

from keras_models.pix2pix import Pix2Pix
from keras_parts.callbacks import ImageSampling
from keras_parts.losses import Pix2PixLosses
from keras_parts.generators import build_generator
from keras_parts.discriminators import build_discriminator
from utils import data


generator = build_generator(input_shape=(256, 256, 3))
discriminator = build_discriminator(input_shape=(256, 256, 3))

g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

g_loss = Pix2PixLosses.loss_g
d_loss = Pix2PixLosses.loss_d

pix2pix = Pix2Pix(generator, discriminator)
pix2pix.compile(
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    g_loss_fn=g_loss,
    d_loss_fn=d_loss,
)


images = tf.random.normal((10, 256, 256, 3))
x = tf.data.Dataset.from_tensor_slices(images).batch(1)
dataset = tf.data.Dataset.zip((x, x))

# callbacks
tensorboard = keras.callbacks.TensorBoard()
image_sampling = ImageSampling(images=dataset.take(5), log_dir='logs')

pix2pix.fit(
    dataset,
    callbacks=[tensorboard, image_sampling],
    epochs=2,
)
