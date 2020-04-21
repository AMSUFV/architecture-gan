import unittest
import tensorflow as tf
from parts import discriminators, generators


class TestForward(unittest.TestCase):

    def test_pix2pix(self):
        gen = generators.pix2pix(input_shape=[512, 512, 3])
        disc = discriminators.pix2pix()

        image = tf.random.normal([1, 512, 512, 3])
        generated = gen(image)
        prediction = disc([generated, image])

        expected_shape = [1, 29, 29, 1]

        self.assertEqual(expected_shape, list(prediction.shape))
