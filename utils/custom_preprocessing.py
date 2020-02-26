import tensorflow as tf

IMG_WIDTH = 512
IMG_HEIGHT = 384

TGT_WIDTH = 512
TGT_HEIGHT = 256

RESIZE_FACTOR = 1.1


def load_images_train(*paths):
    images = load(paths)
    images = random_jitter(images)
    images = normalize(images)
    return images


def load_images_test(*paths):
    images = load(paths)
    images = resize(IMG_WIDTH, IMG_HEIGHT, images)
    images = central_crop(images)
    images = normalize(images)
    return images


def central_crop(images):
    stack = tf.stack(images)
    offset_height = (IMG_HEIGHT - TGT_HEIGHT) // 2
    crop = tf.image.crop_to_bounding_box(stack, offset_height=offset_height, offset_width=0, target_height=TGT_HEIGHT,
                                         target_width=TGT_WIDTH)
    crop = tf.unstack(crop, num=len(images))
    return crop


def load(paths):
    images = []
    for path in paths:
        image = tf.io.read_file(tf.squeeze(path))
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        images.append(image)
    return images


# @tf.function
def random_jitter(images):
    new_width = round(IMG_WIDTH * RESIZE_FACTOR)
    new_height = round(IMG_HEIGHT * RESIZE_FACTOR)

    images = resize(new_width, new_height, images)
    images = random_crop(images)

    if tf.random.uniform(()) > 0.5:
        flipped_images = []
        for image in images:
            flipped_image = tf.image.flip_left_right(image)
            flipped_images.append(flipped_image)
        return flipped_images
    return images


def resize(width, height, images):
    resized_images = []
    for image in images:
        resized_image = tf.image.resize(image, [height, width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_images.append(resized_image)
    return resized_images


def random_crop(images):
    stack = tf.stack(images)
    crop = tf.image.random_crop(stack, size=[len(images), TGT_HEIGHT, TGT_WIDTH, 3])
    crop = tf.unstack(crop, num=len(images))
    return crop


def normalize(images):
    normalized_images = []
    for image in images:
        normalized_image = (image / 127.5) - 1
        normalized_images.append(normalized_image)

    return normalized_images
