import tensorflow as tf
from utils import dataset_tool
from models import pix2pix_updated as pix


class Masker(pix.Pix2Pix):
    def __init__(self, **kwargs):
        self.model = pix.generator(activation='sigmoid')
        super().__init__(gen=self.model, **kwargs)

    def save(self, name):
        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        x = self.model(inputs)
        outputs = tf.keras.layers.Lambda(lambda z: tf.where(z > 0.5, 1.0, 0.0))(x)
        full_model = tf.keras.Model(inputs, outputs)
        full_model.save(name)

    def _log_images(self, dataset, writer, epoch):
        if writer is not None:
            x, _ = next(dataset.take(1).__iter__())

            g_x = self.generator(x, training=False)
            g_x = tf.where(g_x > 0.5, 1.0, 0.0)

            stack = tf.stack([x, g_x], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image('prediction', stack, step=epoch, max_outputs=2)


def mask_train():
    dataset_tool.setup_paths('../dataset')
    temples = ['temple_0', 'temple_2', 'temple_5', 'temple_9']
    train_ds, val_ds = dataset_tool.get_dataset_segmentation(temples, split=0.3, repeat=2, mask=True)

    masker = Masker()

    masker.fit(train_ds, val_ds, epochs=5, log_dir='../logs/masking_0259')
    masker.generator.save('./trained_models/masker_0259.h5')
    masker.save('./trained_models/masker_0259_customsave.h5')


if __name__ == '__main__':
    mask_train()
