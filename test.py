import argparse
import os
import tensorflow as tf

from datetime import datetime
from utils import data

ps = argparse.ArgumentParser()

ps.add_argument('--model_path', required=True)
ps.add_argument('--log_dir', default='logs/test')
ps.add_argument('--samples_path', required=True)
ps.add_argument('--width', type=int, default=512)
ps.add_argument('--height', type=int, default=512)

args = ps.parse_args()

if not os.path.isabs(args.log_dir):
    args.log_dir = os.path.abspath(args.log_dir)

if not os.path.isabs(args.model_path):
    args.model_path = os.path.abspath(args.model_path)

if not os.path.isabs(args.samples_path):
    args.samples_path = os.path.abspath(args.samples_path)

model = tf.keras.models.load_model(args.model_path)
dataset = data.get_simple_dataset(args.width, args.height, args.samples_path + r'\*')

name_pos = args.samples_path.rfind('\\') + 1
folder_name = args.samples_path[name_pos:]
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_path = '/'.join([args.log_dir, model.name, folder_name, time])

writer = tf.summary.create_file_writer(log_path)
with writer.as_default():
    with tf.name_scope('images'):
        for i, x in enumerate(dataset):
            gx = model(x, training=False)
            tf.summary.image('x', tf.squeeze(x, axis=0) * 0.5 + 0.5, step=i)
            tf.summary.image('gx', gx * 0.5 + 0.5, step=i)
