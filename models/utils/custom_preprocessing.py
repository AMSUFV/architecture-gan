import tensorflow as tf

IMG_HEIGHT = 256
IMG_WIDTH = 512
RESIZE_FACTOR = 10


def load_images_train(*paths):
    images = load(paths)



def load(*paths):
    images = []
    for path in paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        images.append(image)
    return images


@tf.function
def random_jitter(*images):
    new_width = IMG_WIDTH + IMG_WIDTH