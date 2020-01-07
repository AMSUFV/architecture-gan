import tensorflow as tf
import matplotlib.pyplot as plt

IMG_HEIGHT = 256
IMG_WIDTH = 512


def load(input_path, real_path):
    paths = [input_path, real_path]
    images = []
    for path in paths:
        # Loading the image
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        images.append(image)
    return images[0], images[1]


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(input_image, real_image):
    # En el codigo original hacen un resize al 111% del tamaño, 256 a 286, se hace lo propio
    # con las dimensiones de nuestra imagen
    resized_width = IMG_WIDTH + IMG_WIDTH // 10
    resized_height = IMG_HEIGHT + IMG_HEIGHT // 10
    # Resize
    input_image, real_image = resize(input_image, real_image, resized_height, resized_width)
    # Random crop
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_images_train(input_path, real_path):
    input_image, real_image = load(input_path, real_path)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_images_test(input_path, real_path):
    input_image, real_image = load(input_path, real_path)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_single_image(path):
    # Loading
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    # Resizing
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Normalizing
    image = (image / 127.5) - 1
    # Adding dims
    image = tf.expand_dims(image, 0)

    return image


def show_image(image, size=(10, 10)):
    plt.figure(figsize=size)
    plt.imshow(image[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.show()


def predict_single_image(model, path, save_path, size=(10, 10), show=False):
    image = load_single_image(path)
    prediction = model(image, training=False)
    if show:
        show_image(prediction, size)
