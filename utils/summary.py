import tensorflow as tf


def summary(metrics_dict, step=None, name='summary'):
    with tf.name_scope(name):
        for name, data in metrics_dict.items():
            tf.summary.scalar(name, data, step=step)
