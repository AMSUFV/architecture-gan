import tensorflow as tf
from models import pix2pix_updated as pix
from utils import dataset_tool


class MaskReconstructor(pix.Pix2Pix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masker = tf.keras.models.load_model('../trained_models/masker_019.h5')

    @tf.function
    def _step(self, train_x, train_y, training=True):
        with tf.GradientTape(persistent=True) as tape:
            m_x = self.masker(train_x, training=False)
            m_y = self.masker(train_y, training=False)
            y = tf.abs(m_y - m_x)

            g_m_x = self.generator(m_x, training=training)

            d_m_y = self.discriminator([m_x, y], training=training)
            d_g_m_x = self.discriminator([m_x, g_m_x], training=training)

            loss_d = self._loss_d(d_m_y, d_g_m_x)
            loss_g, l1_loss = self._loss_g(d_g_x=d_g_m_x, g_x=g_m_x, y=y, multiplier=100)

        if training:
            self._gradient_update(tape, loss_g, self.generator, self.optimizer_g)
            self._gradient_update(tape, loss_d, self.discriminator, self.optimizer_d)

        metrics_names = ['loss_gen_total', 'loss_gen_l1', 'loss_disc']
        metrics = [loss_g, l1_loss, loss_d]
        return metrics_names, metrics

    def fit(self, train, validation, epochs=1, log_dir=None):
        writer_train, writer_val = None, None
        metrics_names, metrics = None, None
        if log_dir is not None:
            writer_train, writer_val = self._get_writers(log_dir)

        for epoch in range(epochs):
            for train_x, train_y in train:
                metrics_names, metrics = self._step(train_x, train_y)
            # self._write_metrics(writer_train, metrics_names, metrics, epoch)
            self._log_images(train, writer_train, epoch)

            for val_x, val_y in validation:
                metrics_names, metrics = self._step(val_x, val_y, training=False)
            # self._write_metrics(writer_val, metrics_names, metrics, epoch)
            self._log_images(validation, writer_val, epoch)

    def _log_images(self, dataset, writer, epoch):
        if writer is not None:
            x, y = next(dataset.take(1).__iter__())
            m_x = self.masker(x, training=False)
            m_y = self.masker(y, training=False)
            tar = m_y - m_x

            g_m_x = self.generator(m_x, training=False)

            tb_images = [x, m_x, tar, g_m_x]
            stack = tf.stack(tb_images, axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('prediction', stack, step=epoch, max_outputs=len(tb_images))


if __name__ == '__main__':
    dataset_tool.setup_paths('../dataset')
    temples = ['temple_0', 'temple_1', 'temple_9']
    train_ds, val_ds = dataset_tool.get_dataset_reconstruction(temples, repeat=2)

    gen = pix.generator(activation='sigmoid')
    pix2pix = MaskReconstructor(gen=gen)
    pix2pix.fit(train_ds, val_ds, epochs=50, log_dir='../logs/test')
    # predictions()
