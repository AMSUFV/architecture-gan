import tensorflow as tf

width = 512
height = 384
resize_factor = 1.3
# Normalization boundaries
a = -1
b = 1

# mask - pink color
R = B = tf.fill((width, height), 1.0)
G = tf.fill((width, height), 0.0)
mask = tf.stack((R, G, B), axis=2)
mask = tf.cast(mask, dtype='float32')
apply_mask = False


def load_images(*paths):
    images = list(map(load, paths))
    images = tf.stack(images)

    if apply_mask:
        images = get_mask(images)

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


def get_mask(images):
    # x being the segmented temple ruins
    # y being the segmented, complete temple
    # z being the true-color temple
    x, y, z = images
    diff = tf.where(x == y, 0, 1)
    # if all the pixel's values are the same, the sum will be 0, eg. [0, 0, 0] vs [1, 0, 1]
    # this gives us a 2D matrix with zeros where the pixels are the same and ones where they are not
    diff = tf.reduce_sum(diff, axis=2)
    diff = tf.expand_dims(diff, axis=2)
    # we keep the real image where they are the same, and put the mask where they differ
    masked = tf.where(diff == 0, z, mask)
    return masked, z
