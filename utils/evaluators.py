class Evaluator:
    def __init__(self, **kwargs):
        self.generator = kwargs.get('generator')
        self.metric = kwargs.get('metric')

    def _test_step(self, data):
        *x, y = data
        gx = self.generator(x, training=False)
        return y, gx

    def _get_metric(self, data):
        return self.metric(*self._test_step(data))

    def evaluate(self, data):
        return [self._get_metric(sample) for sample in data]


class StepEvaluator(Evaluator):
    def __init__(self, segmenter, color_reconstructor, reconstructor, **kwargs):
        super().__init__(**kwargs)
        self.segmenter = segmenter
        self.color_reconstructor = color_reconstructor
        self.reconstructor = reconstructor

    def _test_step(self, data):
        *x, y = data
        x_c = self.segmenter(x, training=False)
        x_cr = self.color_reconstructor(x_c, training=False)
        x_tc = self.reconstructor([x, x_cr], training=False)
        return y, x_tc
