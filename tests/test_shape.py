import unittest
from tensorflow import random
from parts import discriminators, generators


class TestShape(unittest.TestCase):
    def setUp(self) -> None:
        self.width = self.height = 512
        self.batch_size = 1
        self.channels = 3
        self.input_shape = [self.width, self.height, self.channels]
        self.image = random.normal([self.batch_size, self.width, self.height, self.channels])

    def test_pix2pix_disc(self):
        disc = discriminators.pix2pix(input_shape=self.input_shape)
        prediction = disc([self.image, self.image])

        shape = list(prediction.shape)
        expected_shape = [self.batch_size, 29, 29, 1]

        self.assertEqual(shape, expected_shape)
