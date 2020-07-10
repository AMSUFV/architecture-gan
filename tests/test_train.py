import os
import tempfile
import shutil
import settings
import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

from utils import data
from keras_parts import builder


class TestTrain(unittest.TestCase):
    def setUp(self):
        # dir creation
        self.dir_logs = tempfile.TemporaryDirectory(dir=os.getcwd())
        self.dir_dataset = tempfile.TemporaryDirectory(dir=os.getcwd())
        self.dir_temples = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)
        self.dir_temples_ruins = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)
        self.dir_temples_colors = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)
        self.dir_temples_ruins_colors = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)

        # assignment
        data.PATH_TEMPLES = self.dir_temples.name
        data.PATH_TEMPLES_RUINS = self.dir_temples_ruins.name
        data.PATH_TEMPLES_COLORS = self.dir_temples_colors.name
        data.PATH_TEMPLES_RUINS_COLORS = self.dir_temples_ruins_colors

        # settings common to all tests
        settings.DATASET_DIR = self.dir_dataset.name
        settings.LOG_DIR = self.dir_logs.name
        settings.EPOCHS = 1
        settings.TEMPLES = [0]
        settings.BUFFER_SIZE = 5

        # images
        image = tf.ones((settings.IMG_WIDTH, settings.IMG_HEIGHT, 3))

        dirs = [self.dir_temples, self.dir_temples_ruins, self.dir_temples_colors, self.dir_temples_ruins_colors]
        temple_dirs = [tempfile.TemporaryDirectory(dir=x.name, prefix='temple_0') for x in dirs]
        for temple_dir in temple_dirs:
            for i in range(settings.BUFFER_SIZE):
                plt.imsave(temple_dir.name + f'\\temple_0_{i}.png', image.numpy())

    @unittest.skip
    def test_reconstruction(self):
        settings.TRAINING = 'reconstruction'
        import keras_train

    @unittest.skip
    def test_color_assisted(self):
        settings.TRAINING = 'color_assisted'
        import keras_train

    # def tearDown(self):
    #     shutil.rmtree(self.dir_dataset.name)
    #     shutil.rmtree(self.dir_logs.name)
