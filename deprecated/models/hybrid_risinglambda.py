import tensorflow as tf
from deprecated.models.hybrid_reconstuctor import HybridReconstructor


class RaisingLamdba(HybridReconstructor):
    def __init__(self, lambda_i=100, lambda_f=200, **kwargs):
        super().__init__(**kwargs)

        # Raising lambda
        # initial and final lambdas
        self.lambda_i = lambda_i
        self.lambda_f = lambda_f
        # current epoch and total epochs
        self.e_t = 0
        self.e_f = None

    def generator_loss(self, disc_generated_output, gen_output, target):
        g_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # Rising lambda
        lambda_t = self.lambda_i + (self.lambda_f - self.lambda_i) / self.e_f * self.e_t
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = g_loss + lambda_t * l1_loss

        return total_gen_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def fit(self, train_ds, test_ds=None, epochs=100):
        self.e_f = epochs
        for epoch in range(epochs):
            self.e_t = epoch + 1
            # Train
            for ruin, temple, color in train_ds:
                self.train_step(ruin, temple, color)

            if test_ds is not None:
                self.validate(test_ds)

            self._metric_update(train_ds, test_ds, epoch)
            if self.log_dir is not None:
                self._train_predict(train_ds, self.train_summary_writer, epoch, 'train')
                if test_ds is not None:
                    self._train_predict(test_ds, self.val_summary_writer, epoch, 'validation')


