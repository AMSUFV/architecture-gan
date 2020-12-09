from models.pix2pix import Pix2Pix
from parts.discriminators import pix2pix_discriminator


class Evaluator:
    def __init__(self, **kwargs):
        self.generator = kwargs.get('generator')
        self.metric = kwargs.get('metric')

    def _test_step(self, data):
        *x, y = data
        gx = self.generator(x, training=False)
        return y, gx

    def _get_metric(self, data):
        return self.metric(*self._test_step(data))

    def evaluate(self, data):
        return [self._get_metric(sample) for sample in data]


class StepEvaluator(Evaluator):
    def __init__(self, segmenter, color_reconstructor, reconstructor, **kwargs):
        super().__init__(**kwargs)
        self.segmenter = segmenter
        self.color_reconstructor = color_reconstructor
        self.reconstructor = reconstructor

    def _test_step(self, data):
        *x, y = data
        x_c = self.segmenter(x, training=False)
        x_cr = self.color_reconstructor(x_c, training=False)
        x_tc = self.reconstructor([x, x_cr], training=False)
        return y, x_tc


# noinspection PyAttributeOutsideInit
class C2ST(Pix2Pix):
    def __init__(self, generator):
        super().__init__(generator, pix2pix_discriminator())

    def train_step(self, images):
        *x, y = images
        gx = self.generator(x, training=False)
        if type(x) == list:
            x = x[0]
        d_loss = self.train_d(x, gx, y)
        return {'d_loss': d_loss}
# class C2ST:
#     def __init__(self, model, discriminator, loss_fn):
#         self.model = model
#         self.discriminator = discriminator
#         self.loss_fn = loss_fn
#
#     def _train_step(self, data):
#         *x, y = data
#         gx = self.model(x, training=False)
#         with tf.GradientTape() as t:
#             dy = self.discriminator([x[0], y], training=True)
#             dgx = self.discriminator([x[0], gx], traning=True)
#             d_loss = self.loss_fn(dy, dgx)
