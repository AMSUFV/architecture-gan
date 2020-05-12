import tensorflow as tf

width, height = 512, 384
resize_factor = 1.3
# Normalization boundaries
a, b = -1, 1

# mask - pink color
R = B = tf.fill((height, width), 255)
G = tf.fill((height, width), 0)
mask = tf.stack((R, G, B), axis=2)
mask = tf.cast(mask, dtype='float32')
apply_mask = False
demasking = False

# image decoding function
img_decoding = tf.io.decode_png


def setup(img_format):
    global img_decoding
    if img_format == 'png':
        img_decoding = tf.io.decode_png
    elif img_format == 'jpeg':
        img_decoding = tf.io.decode_jpeg


def load_images(*paths):
    images = list(map(load, paths))
    images = tf.stack(images)
    images = jitter(images)
    if apply_mask:
        images = get_mask(images)
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
    if tf.random.uniform(()) > 0.5:
        return tf.image.flip_left_right(cropped)
    return cropped


def resize(image):
    return tf.image.resize(image, [int(height * resize_factor), int(width * resize_factor)],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def random_crop(images):
    return tf.image.random_crop(images, size=[images.shape[0], height, width, 3])


def get_mask(images):
    if demasking:
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
    masked_temple = tf.where(diff == 0, temple, mask)
    if demasking:
        return tf.stack([masked_temple, temple])
    else:
        return tf.stack([ruins, masked_temple])
