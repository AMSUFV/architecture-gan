import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks


class ImageSampling(callbacks.Callback):
    def __init__(self, train_images, val_images, frequency, log_dir):
        super(ImageSampling).__init__()
        self.data = [
            (train_images, "train"),
            (val_images, "validation"),
        ]
        self.frequency = frequency
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:

            for images, scope in self.data:
                predictions = []
                for image in images:
                    if isinstance(image, tuple):
                        image = image[0]
                    predictions.append(
                        self.model.generator(image, training=False) * 0.5 + 0.5
                    )
                predictions = tf.squeeze(predictions)
                with tf.name_scope(scope):
                    with self.writer.as_default():
                        tf.summary.image(
                            name=f"image",
                            data=predictions,
                            step=epoch,
                            max_outputs=predictions.shape[0],
                        )
