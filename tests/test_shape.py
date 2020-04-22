import unittest
from tensorflow import random
from parts import discriminators, generators


class TestShape(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 1
        self.image = random.normal((self.batch_size, 512, 512, 3))

    def test_pix2pix_disc(self):
        disc = discriminators.pix2pix(input_shape=self.image.shape[1:])
        prediction = disc([self.image, self.image])

        shape = list(prediction.shape)
        expected_shape = [self.batch_size, 29, 29, 1]

        self.assertEqual(shape, expected_shape)

    def test_pix2pix_gen(self):
        gen = generators.pix2pix(input_shape=self.image.shape[1:])

        generated = gen(self.image)
        self.assertEqual(self.image.shape, generated.shape)

    def test_resnet_gen(self):
        gen = generators.resnet(input_shape=self.image.shape[1:])
        generated = gen(self.image)

        self.assertEqual(self.image.shape, generated.shape)
