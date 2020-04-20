import unittest
from tensorflow import random
from parts import discriminators, generators


class TestShape(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 1

        self.input_shape_256 = [256, 256, 3]
        self.image_256 = random.normal([self.batch_size, 256, 256, 3])

        self.input_shape_512 = [512, 512, 3]
        self.image_512 = random.normal([self.batch_size, 512, 512, 3])

    def test_pix2pix_disc(self):
        disc = discriminators.pix2pix(input_shape=self.input_shape_512)
        prediction = disc([self.image_512, self.image_512])

        shape = list(prediction.shape)
        expected_shape = [self.batch_size, 29, 29, 1]

        self.assertEqual(shape, expected_shape)

    def test_pix2pix_gen(self):
        gen = generators.pix2pix()

        generated = gen(self.image_512)
        self.assertEqual(self.image_512.shape, generated.shape)

    def test_resnet_gen(self):
        gen = generators.resnet()
        generated = gen(self.image_256)

        self.assertEqual(self.image_256.shape, generated.shape)
