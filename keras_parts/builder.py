import keras_models
import tensorflow as tf
from tensorflow.keras import optimizers
from keras_parts import generators, discriminators, losses


def get_model(name, training, input_shape, norm_type="batch"):
    if name == "pix2pix":
        discriminator = discriminators.pix2pix_discriminator(
            input_shape, norm_type=norm_type
        )
        if training == "color_assisted":
            generator = generators.pix2pix_generator(
                input_shape, assisted=True, norm_type=norm_type
            )
            model = keras_models.Assisted
        else:
            generator = generators.pix2pix_generator(
                input_shape, assisted=False, norm_type=norm_type
            )
            model = keras_models.Pix2Pix
        model = model(generator, discriminator)

        g_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        d_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        model.compile(
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            g_loss_fn=losses.Pix2PixLosses.loss_g,
            d_loss_fn=losses.Pix2PixLosses.loss_d,
        )

        return model
