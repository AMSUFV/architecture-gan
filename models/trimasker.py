import tensorflow as tf
from models.pix2pix_updated import Pix2Pix
from utils import dataset_tool
import matplotlib.pyplot as plt


class MaskReconstructor(Pix2Pix):
    def __init__(self):
        super().__init__()
        self.masker = tf.keras.models.load_model('../trained_models/masker_019.h5')

    @tf.function
    def _step(self, train_x, train_y, training=True):
        with tf.GradientTape(persistent=True) as tape:
            g_m_x = self.generator(train_x, training=training)

            d_m_y = self.discriminator([train_x, train_y], training=training)
            d_g_m_x = self.discriminator([train_x, g_m_x], training=training)

            loss_d = self._loss_d(d_m_y, d_g_m_x)
            loss_g, l1_loss = self._loss_g(d_g_x=d_g_m_x, g_x=g_m_x, y=train_y, multiplier=100)

        if training:
            # self._gradient_update(tape, loss_g, self.generator, self.optimizer_g)
            # self._gradient_update(tape, loss_d, self.discriminator, self.optimizer_d)
            gradients_g = tape.gradient(loss_g, self.generator.trainable_variables)
            gradients_d = tape.gradient(loss_d, self.discriminator.trainable_variables)
            self.optimizer_g.apply_gradients(zip(gradients_g, self.generator.trainable_variables))
            self.optimizer_d.apply_gradients(zip(gradients_d, self.discriminator.trainable_variables))

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
                m_x = self.masker(train_x, training=False)
                m_y = self.masker(train_y, training=False)
                self._step(m_x, m_y)
            self._log_images(train, writer_train, epoch)

            for val_x, val_y in validation:
                m_x = self.masker(val_x, training=False)
                m_y = self.masker(val_y, training=False)
                self._step(m_x, m_y, training=False)
            self._log_images(validation, writer_val, epoch)

    def _log_images(self, dataset, writer, epoch):
        if writer is not None:
            x, y = next(dataset.take(1).__iter__())

            m_x = self.masker(x, training=False)
            m_y = self.masker(y, training=False)
            g_m_x = self.generator(m_x, training=False)

            tb_images = [m_x, g_m_x]
            stack = tf.stack(tb_images, axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('prediction', stack, step=epoch, max_outputs=len(tb_images))


def predictions():
    model = tf.keras.models.load_model('../trained_models/masker_019.h5')
    dataset_tool.setup_paths('../dataset')
    pred_tr, pred_val = dataset_tool.get_dataset_reconstruction(['temple_0'], repeat=2)
    writer = tf.summary.create_file_writer('../logs/test/masker_test')
    for i, (image, _) in enumerate(pred_tr):
        prediction = model(image, training=False) * 0.5 + 0.5
        # prediction = tf.squeeze(prediction)
        with writer.as_default():
            tf.summary.image('prediction', prediction, i)


if __name__ == '__main__':
    dataset_tool.setup_paths('../dataset')
    train_ds, val_ds = dataset_tool.get_dataset_reconstruction(['temple_0'], repeat=2)
    pix2pix = Pix2Pix()
    pix2pix.fit(train_ds, val_ds, epochs=10, log_dir='../logs/test')
