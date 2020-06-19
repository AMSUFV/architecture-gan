import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


bce = keras.losses.BinaryCrossentropy(from_logits=True)


def loss_g(y, gx, dgx):
    loss_dgx = bce(tf.ones_like(dgx), dgx)
    loss_l1 = tf.reduce_mean(tf.abs(y - gx))
    total_loss = loss_dgx + LAMBDA * loss_l1
    return total_loss, loss_l1


def loss_d(dy, dgx):
    loss_y = bce(tf.ones_like(dy), dy)
    loss_gx = bce(tf.zeros_like(dgx), dgx)
    return (loss_y + loss_gx) / 2


"""
## Prepare the dataset
"""

# Variables
BUFFER_SIZE = 400
BATCH_SIZE = 1
LAMBDA = 100
WIDTH = HEIGHT = 256
CHANNELS = 3

url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"
path = keras.utils.get_file("facades.tar.gz", origin=url, extract=True)
path = os.path.join(os.path.dirname(path), "facades/")

train = tf.data.Dataset.list_files(path + "train/*.jpg")
train = train.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val = tf.data.Dataset.list_files(path + "val/*.jpg")
val = val.map(load_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val = val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


"""
## Train the end-to-end model
"""

generator = build_generator()
discriminator = build_discriminator()
generator.summary()
discriminator.summary()


pix2pix = Pix2Pix(generator, discriminator)
pix2pix.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
    d_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
    g_loss_fn=loss_g,
    d_loss_fn=loss_d,
)

pix2pix.fit(train, epochs=5, validation_data=val)
