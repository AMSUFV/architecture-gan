import unittest
import tensorflow as tf
from models.pix2pix_variants import Pix2Pix
from models.resnet import ResNet


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.images = tf.random.normal((4, 512, 512, 3))
        x = tf.data.Dataset.from_tensor_slices(self.images).batch(1)
        y = tf.data.Dataset.from_tensor_slices(self.images).batch(1)
        self.dataset = tf.data.Dataset.zip((x, y))

    def test_pix2pix(self):
        pix2pix = Pix2Pix(input_shape=self.images.shape[1:])
        pix2pix.fit(dataset=self.dataset, epochs=1, path='')

    def test_resnet(self):
        resnet = ResNet(input_shape=self.images.shape[1:])
        resnet.fit(dataset=self.dataset, epochs=1, path='')
