import tensorflow as tf

WIDTH, HEIGHT = 512, 512
RESIZE_FACTOR = 1.3

# Normalization boundaries
A, B = -1, 1

# mask - pink color
MASK = None
APPLY_MASK = False
DEMASKING = False


def get_mask():
    red = blue = tf.fill((HEIGHT, WIDTH), 255)
    green = tf.fill((HEIGHT, WIDTH), 0)
    mask = tf.stack((red, green, blue), axis=2)
    return tf.cast(mask, dtype='float32')


def load_images(*paths):
    images = list(map(load, paths))
    images = tf.stack(images)
    images = jitter(images)
    if APPLY_MASK:
        images = mask_image(images)
    # feature scaling assuming max will always be 255 and min will always be 0 for all images
    images = A + (images * (B - A)) / 255
    return tf.unstack(images, num=images.shape[0])


def load_test_images(*paths):
    images = list(map(load, paths))
    images = resize(tf.stack(images), HEIGHT, WIDTH)
    return tf.unstack(images, num=images.shape[0])


def load(path):
    file = tf.io.read_file(tf.squeeze(path))
    image = tf.io.decode_png(file, channels=3)
    return tf.cast(image, tf.float32)


def jitter(images):
    # Resizes the images to the closes size to the target size without losing aspect ratio
    # resized = resize_nearest_size(images)
    # Prepares the images for the random crop
    # resized = resize(resized, int(images[0].shape[0] * RESIZE_FACTOR), int(images[0].shape[1] * RESIZE_FACTOR))
    cropped = random_crop(images)
    if tf.random.uniform(()) > 0.5:
        return tf.image.flip_left_right(cropped)
    return cropped


def resize_nearest_size(images):
    img_height, img_width = images[0].shape[0], images[0].shape[1]
    ratio = img_width / WIDTH
    return resize(images, int(img_height / ratio), int(img_width / ratio))


def resize(image, h, w):
    return tf.image.resize(image, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def random_crop(images):
    return tf.image.random_crop(images, size=[images.shape[0], HEIGHT, WIDTH, 3])


def mask_image(images):
    if DEMASKING:
        seg_ruin, seg_temple, temple = tf.unstack(images, num=images.shape[0])
        ruins = None
    else:
        seg_ruin, seg_temple, temple, ruins = tf.unstack(images, num=images.shape[0])

    diff = tf.where(seg_ruin == seg_temple, 0, 1)
    # if all the pixel's values are the same, the sum will be 0, eg. [0, 0, 0] vs [1, 0, 1]
    # this gives us a 2D matrix with zeros where the pixels are the same and ones where they are not
    diff = tf.reduce_sum(diff, axis=2)
    diff = tf.expand_dims(diff, axis=2)
    # we keep the real image where they are the same, and put the mask where they differ
    masked_temple = tf.where(diff == 0, temple, MASK)
    if DEMASKING:
        return tf.stack([masked_temple, temple])
    else:
        return tf.stack([ruins, masked_temple])
