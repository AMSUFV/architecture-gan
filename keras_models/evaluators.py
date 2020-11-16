import tensorflow as tf


class L2Evaluator:
    def __init__(self, generator, assisted=False):
        self.generator = generator
        self.assisted = assisted

    def _test_step(self, data):
        *x, y = data
        gx = self.generator(x, training=False)
        l2_distance = tf.math.sqrt(tf.reduce_sum((y - gx)**2))
        return l2_distance

    def evaluate(self, data):
        results = [self._test_step(sample).numpy() for sample in data]
        return results
