import tensorflow as tf

class Metric:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __call__(self, y, gx):
        return self.value(y, gx)

MSE = Metric('mean_squared_error', lambda y, gx: tf.math.reduce_mean((y - gx) ** 2).numpy())
RMSE = Metric('root_mean_squared_error', lambda y, gx: tf.math.sqrt(MSE(y, gx)).numpy())
PSNR = Metric('peak_signal_to_noise_ratio', lambda y, gx: tf.image.psnr(y, gx, 255).numpy())
SSIM = Metric('structural_similarity_index_measure', lambda y, gx: tf.image.ssim(y, gx, 255).numpy())

