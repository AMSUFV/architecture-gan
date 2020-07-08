import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks


class ImageSampling(callbacks.Callback):
    def __init__(self, images, log_dir):
        super(ImageSampling).__init__()
        self.images = images
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        for image in self.images:
            predictions.append(self.model.generator(image) * 0.5 + 0.5)
        predictions = tf.squeeze(predictions)
        with self.writer.as_default():
            tf.summary.image(
                name="images",
                data=predictions,
                step=epoch,
                max_outputs=predictions.shape[0],
            )


class Validation(callbacks.Callback):
    def __init__(self, validation_data, log_dir):
        super(Validation).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        pass
