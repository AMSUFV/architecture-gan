import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

from utils import data


@unittest.skip('Unfinished test')
class TestTrainNew(unittest.TestCase):
    def setUp(self) -> None:
        # dir creation
        self.dir_dataset = tempfile.TemporaryDirectory(dir=os.getcwd())
        self.dir_temples = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)
        self.dir_temples_ruins = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)
        self.dir_temples_colors = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)
        self.dir_temples_ruins_colors = tempfile.TemporaryDirectory(dir=self.dir_dataset.name)

        # name assignment
        data.PATH_TEMPLES = self.dir_temples.name
        data.PATH_TEMPLES_RUINS = self.dir_temples_ruins.name
        data.PATH_TEMPLES_COLORS = self.dir_temples_colors.name
        data.PATH_TEMPLES_RUINS_COLORS = self.dir_temples_ruins_colors

        # images
        width = height = 512
        image = tf.ones((width, height, 3))

        dirs = [self.dir_temples, self.dir_temples_ruins, self.dir_temples_colors, self.dir_temples_ruins_colors]
        temple_dirs = [tempfile.TemporaryDirectory(dir=x.name) for x in dirs]
        for temple_dir in temple_dirs:
            index = temple_dir.name.rfind('\\') + 1
            temple_dir.name = temple_dir.name[index] + 'temple_0'
            plt.imsave(temple_dir.name + '\\temple_0.png')
