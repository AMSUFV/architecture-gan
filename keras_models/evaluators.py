import tensorflow as tf


class L2Evaluator:
    def __init__(self, generator):
        self.generator = generator

    def _test_step(self, data):
        *x, y = data
        gx = self.generator(x, training=False)
        if type(x) == list:
            x = x[0]
        l2_distance = tf.math.sqrt(tf.reduce_sum((x - gx)**2))
        return l2_distance

    def evaluate(self, data):
        results = [self._test_step(sample).numpy() for sample in data]
        return results
