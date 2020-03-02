import tensorflow as tf
from models.pix2pix import Pix2Pix


class Trimodel:
    def __init__(self):
        model_source = Pix2Pix()

        # Segmenter
        self.g_segmenter = model_source.build_generator()
        self.d_segmenter = model_source.build_discriminator()

        # From segmented ruins to segmented reconstructions
        self.g_color_reconstructor = model_source.build_generator()
        self.d_color_ = model_source.build_discriminator()

        # Colored reconstruction
        self.g_real_reconstructor = model_source.build_generator(heads=2)
        self.g_real_discriminator = model_source.build_discriminator()


if __name__ == '__main__':
    trimodel = Trimodel()
