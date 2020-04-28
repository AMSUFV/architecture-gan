from parts.generators import resnet as resnet_generator
from models.pix2pix_variants import Pix2Pix
from parts import losses


class ResNet(Pix2Pix):
    def __init__(self, input_shape=(None, None, 3), norm_type='instancenorm'):
        super().__init__(input_shape, norm_type)
        self.name = 'resnet'
        self.generator = resnet_generator(input_shape)
        self.loss_d, self.loss_g = losses.pix2pix()
