import argparse
import os
from utils import get_model, get_dataset, setup_paths
from utils import preprocessing
from utils import data
import tensorflow as tf

ps = argparse.ArgumentParser()

training = ps.add_argument_group('training')
training.add_argument('--model', default='pix2pix')
training.add_argument('--epochs', type=int, default=100)
training.add_argument('--log_dir', default='logs/')

ds = ps.add_argument_group('dataset', 'dataset configuration settings')
ds.add_argument('--training_type', default='reconstruction', choices=['color_assisted',
                                                                      'color_reconstruction',
                                                                      'reconstruction',
                                                                      'segmentation',
                                                                      'de-segmentation',
                                                                      'masking',
                                                                      'de-masking'
                                                                      ])
ds.add_argument('--dataset_dir', default='dataset/')
ds.add_argument('--temples', type=int, nargs='+')
ds.add_argument('--split', type=float, default=0.25)
ds.add_argument('--batch_size', type=int, default=1)
ds.add_argument('--buffer_size', type=int, default=400)
ds.add_argument('--repeat', type=int, default=1)
ds.add_argument('--img_format', default='png')
ds.add_argument('--img_width', type=int, default=512)
ds.add_argument('--img_height', type=int, default=384)

args = ps.parse_args()

img_format = args.img_format.strip('.').lower()
preprocessing.setup(img_format)

if args.training_type in ['color_assisted', 'de-masking']:
    data.repetitions = [1, args.repeat, args.repeat]
elif args.training_type == 'masking':
    data.repetitions = [1, args.repeat, args.repeat, 1]
else:
    data.repetitions = [1, args.repeat]

if not os.path.isabs(args.log_dir):
    args.log_dir = os.path.abspath(args.log_dir)

ds_args = [args.temples, args.split, args.batch_size, args.buffer_size]
train, val = data.get_dataset(args.dataset_dir, args.training_type, *ds_args)

writer = tf.summary.create_file_writer(f'logs/test/{args.training_type}')
with writer.as_default():
    for i, (x, y) in enumerate(train):
        x_norm = (x + 1) / 2
        y_norm = (y + 1) / 2
        tf.summary.image('x', x, step=i)
        tf.summary.image('y', y, step=i)


# model = get_model(args.model, args.training_type)
#
# log_path = os.path.join(os.getcwd(), args.log_dir)
# model.fit(train, args.epochs, path=log_path)
