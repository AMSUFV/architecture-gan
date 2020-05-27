import tensorflow as tf
from deprecated.utils import dataset_tool
from deprecated.models import pix2pix_updated as pix


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


class MaskReconstructor(pix.Pix2Pix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masker = tf.keras.models.load_model('../trained_models/masker_0259_customsave.h5')

    # def _loss_g(self, d_g_x, **kwargs):
    #     loss_d_g_x = self.loss_object(tf.ones_like(d_g_x), d_g_x)
    #     # difference between the number of zeros (or mask size) between y and g_x
    #     loss_diff = kwargs['multiplier'] * tf.abs(1 - tf.reduce_sum(kwargs['y']) / tf.reduce_sum(kwargs['g_x']))
    #     loss_total = loss_d_g_x + loss_diff
    #     return loss_total, loss_diff
    # todo: implement area loss
    
    def fit(self, train, validation, epochs=1, log_dir=None):
        writer_train, writer_val = None, None
        if log_dir is not None:
            writer_train, writer_val = self._get_writers(log_dir)

        for epoch in range(epochs):
            for dataset, writer, training in zip([train, validation], [writer_train, writer_val], [True, False]):
                for x, y in dataset:
                    m_x = self.masker(x, training=False)
                    m_y = self.masker(y, training=False)
                    missing = tf.abs(m_y - m_x)
                    x_missing = tf.where(missing == 0.0, x, -1.0)

                    self._step(x_missing, y, epoch, training, writer)

                self._log_images(dataset, writer, epoch)

    def _log_images(self, dataset, writer, epoch):
        if writer is not None:
            x, y = next(dataset.take(1).__iter__())
            m_x = self.masker(x, training=False)
            m_y = self.masker(y, training=False)
            missing = tf.abs(m_y - m_x)
            x_missing = tf.where(missing == 0, x, -1.0)

            g_x = self.generator(x_missing, training=False)

            stack = tf.stack([x, x_missing, y, g_x], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('prediction', stack, step=epoch, max_outputs=4)


def train_mask():
    dataset_tool.setup_paths('../dataset')
    temples = ['temple_0', 'temple_2', 'temple_5', 'temple_9']
    train_ds, val_ds = dataset_tool.get_dataset_segmentation(temples, split=0.3, repeat=2, mask=True)

    masker = Masker()

    masker.fit(train_ds, val_ds, epochs=5, log_dir='../logs/masking_0259')
    masker.generator.save('../trained_models/masker_0259.h5')
    masker.save('../trained_models/masker_0259_customsave.h5')


def train_mask_reconstruction():
    dataset_tool.setup_paths('../dataset')
    temples = ['temple_0', 'temple_2', 'temple_5', 'temple_9']
    train_ds, val_ds = dataset_tool.get_dataset_reconstruction(['temple_0'], split=0.3, repeat=2)

    mask_reconstructor = MaskReconstructor()

    mask_reconstructor.fit(train_ds, val_ds, epochs=50, log_dir='../logs/missingarea_0')
    mask_reconstructor.generator.save('../trained_models/missingarea_0.h5')


if __name__ == '__main__':
    train_mask_reconstruction()
