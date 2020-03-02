from models.pix2pix import Pix2Pix
import tensorflow as tf


class RisingLambda(Pix2Pix):
    def __init__(self, lambda_i=100, lambda_f=200, **kwargs):
        super().__init__(**kwargs)

        self.lambda_i = lambda_i
        self.lambda_f = lambda_f

        self.e_t = 0
        self.e_f = None

    def generator_loss(self, disc_generated_output, gen_output, target):
        g_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # Rising lambda
        lambda_t = self.lambda_i + (self.lambda_f - self.lambda_i) / self.e_f * self.e_t
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = g_loss + lambda_t * l1_loss

        return total_gen_loss

    def fit(self, train_ds, test_ds=None, epochs=50):
        self.e_f = epochs
        for epoch in range(epochs):
            self.e_t += 1
            for ruin, temple in train_ds:
                self.train_step(ruin, temple)
            self._metric_update(train_ds, test_ds, epoch)
            self._train_predict(train_ds, self.train_summary_writer, epoch, 'train')

            if test_ds is not None:
                self.validate(test_ds)
                self._train_predict(test_ds, self.val_summary_writer, epoch, 'validation')
