import tensorflow as tf

from parts.generators import resnet as generator
from models.pix2pix_variants import Pix2Pix
from parts import losses


class ResNet(Pix2Pix):
    def __init__(self, input_shape=(512, 512, 3)):
        super().__init__(input_shape)
        self.generator = generator(input_shape)
        self.loss_d, self.loss_g = losses.pix2pix()
