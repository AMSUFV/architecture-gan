import argparse
import tensorflow as tf
from utils import dataset_tool
#
# parser = argparse.ArgumentParser(description='Launch model prediction')
#
# parser.add_argument('-model', '-m', type=str, required=True,
#                     help='Model to use for the prediction(s).')
# parser.add_argument('-source', '-s', type=str, required=True,
#                     help='What to predict. Path to a single image or to an image folder')

segmenter = tf.keras.models.load_model('trained_models/segmenter.h5')
reconstructor_color = tf.keras.models.load_model('trained_models/reconstruction_color_pix2pix.h5')
hybrid = tf.keras.models.load_model('trained_models/colors_all0_risinglambda.h5')

dataset_tool.setup_paths('dataset')
temple = 'temple_5'
ruins = 'ruins_1'
dataset = dataset_tool.get_dataset_prediction(f'dataset/temples_ruins/{temple}_{ruins}')

log_path = f'logs/predict/3models/{temple}_{ruins}'
writer = tf.summary.create_file_writer(log_path)

for i, image in enumerate(dataset):
    segmented = segmenter(image, training=False)
    reconstructed_color = reconstructor_color(segmented, training=False)
    reconstructed_real = hybrid([image, reconstructed_color], training=False)
    image = tf.squeeze(image, axis=0)
    stack = tf.stack([image, segmented, reconstructed_color, reconstructed_real], axis=0) * 0.5 + 0.5
    stack = tf.squeeze(stack)

    with writer.as_default():
        tf.summary.image(f'{temple}_{ruins}', stack, step=i, max_outputs=4)
