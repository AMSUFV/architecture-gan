import unittest
import tensorflow as tf
from models.new_pix2pix import Pix2Pix


class TestPix2Pix(unittest.TestCase):
    def setUp(self) -> None:
        self.pix2pix = Pix2Pix()
        images = tf.random.normal([4, 512, 512, 3])
        x = tf.data.Dataset.from_tensor_slices(images).batch(1)
        y = tf.data.Dataset.from_tensor_slices(images).batch(1)
        self.dataset = tf.data.Dataset.zip((x, y))

    def test_fit(self):
        self.pix2pix.fit(dataset=self.dataset, epochs=1, path='')
