import unittest
import tensorflow as tf
from models import ResNet


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        image = tf.random.normal((5, 512, 512, 3))
        x, y = tf.data.Dataset.from_tensor_slices(image)
        self.dataset = tf.data.Dataset.zip((x, y)).batch(1)

    def test_resnet(self):
        resnet = ResNet(input_shape=(512, 512, 3))
        resnet.fit(self.dataset, 1)
