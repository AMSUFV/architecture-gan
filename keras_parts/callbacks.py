import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks


class ImageSampling(callbacks.Callback):
    def __init__(self, train_images, val_images, frequency, log_dir):
        super(ImageSampling).__init__()
        self.train_images = train_images
        self.val_images = val_images
        self.frequency = frequency
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            data = [
                (self.train_images, "train"),
                (self.val_images, "validation"),
            ]

            for images, scope in data:
                predictions = []
                for i, image in enumerate(images):
                    predictions.append(self.model.generator(image) * 0.5 + 0.5)
                predictions = tf.squeeze(predictions)

                with tf.name_scope(scope):
                    with self.writer.as_default():
                        tf.summary.image(
                            name=f"image_{i}",
                            data=predictions,
                            step=epoch,
                            max_outputs=predictions.shape[0],
                        )
