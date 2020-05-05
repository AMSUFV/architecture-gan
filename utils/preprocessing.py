import tensorflow as tf

width = 512
height = 384
resize_factor = 1.3
# Normalization boundaries
a = -1
b = 1


def load_images(*paths):
    images = list(map(load, paths))
    images = tf.stack(images)
    images = jitter(images)
    # feature scaling assuming max will always be 255 and min will always be 0 for all images
    images = a + (images * (b - a)) / 255
    return tf.unstack(images, num=images.shape[0])


def load(path):
    file = tf.io.read_file(tf.squeeze(path))
    image = tf.io.decode_png(file, channels=3)
    return tf.cast(image, tf.float32)


def jitter(images):
    resized = resize(images)
    cropped = random_crop(resized)
    return tf.image.random_flip_left_right(cropped)


def resize(image):
    return tf.image.resize(image, [int(height * resize_factor), int(width * resize_factor)],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def random_crop(images):
    return tf.image.random_crop(images, size=[images.shape[0], width, height, 3])
