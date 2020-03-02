import tensorflow as tf
from utils import dataset_creator
from models.hybrid_risinglambda import RaisingLamdba


class FaultyRec(RaisingLamdba):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color_reconstructor = tf.keras.models.load_model('../trained_models/reconstruction_color_pix2pix.h5')

    def fit(self, train_ds, val_ds=None, epochs=100):
        self.e_f = epochs
        for epoch in range(epochs):
            self.e_t = epoch + 1
            # Train
            for ruin, color, temple, ruin_color in train_ds:
                if tf.random.uniform(()) > 0.5:
                    color = self.color_reconstructor(ruin_color, training=False)
                self.train_step(ruin, temple, color)

            if val_ds is not None:
                self.validate(val_ds)

            self._metric_update(train_ds, val_ds, epoch)
            if self.log_dir is not None:
                for ruin, color, _, _ in train_ds.take(1):
                    inputs = [ruin, color]
                    self._log_prediction(inputs, self.train_summary_writer, epoch)

                if val_ds is not None:
                    for ruin, _, _, ruin_color in val_ds.take(1):
                        color = self.color_reconstructor(ruin_color, training=False)
                        inputs = [ruin, color]
                        self._log_prediction(inputs, self.val_summary_writer, epoch)

    def validate(self, validation_dataset):
        for val_ruin, _, val_temple, val_ruin_color in validation_dataset:
            val_color = self.color_reconstructor(val_ruin_color, training=False)

            gen_output = self.generator([val_ruin, val_color], training=False)

            disc_real_output = self.discriminator([val_ruin, val_temple], training=False)
            disc_generated_output = self.discriminator([val_ruin, gen_output], training=False)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, val_temple)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            if self.log_dir is not None:
                self.val_disc_loss(disc_loss)
                self.val_gen_loss(gen_loss)
                self.val_gen_acc(tf.zeros_like(disc_generated_output), disc_generated_output)
                self.val_real_acc(tf.ones_like(disc_real_output), disc_real_output)

    def _log_prediction(self, inputs, writer, step):
        prediction = self.generator(inputs, training=False) * 0.5 + 0.5
        input_stack = tf.stack(inputs, axis=0) * 0.5 + 0.5
        input_stack = tf.squeeze(input_stack)

        with writer.as_default():
            tf.summary.image('inputs', input_stack, step=step, max_outputs=len(inputs))
            tf.summary.image('prediction', prediction, step=step)


if __name__ == '__main__':
    log_dir = '../logs/test'

    temples = ['temple_1', 'temple_5', 'temple_6', 'temple_9']
    dataset_creator.setup_paths('../dataset')
    train, val = dataset_creator.get_dataset_combined(temples, repeat=2)

    hybrid = FaultyRec(log_dir=log_dir, autobuild=False)
    hybrid.build_discriminator(inplace=True)
    hybrid.build_generator(heads=2, inplace=True)

    hybrid.fit(train, val, epochs=50)
    hybrid.generator.save('../trained_models/colors_1569_faulty.h5')
