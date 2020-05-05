import argparse
import os
from utils import get_model, get_dataset, setup_paths

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
                                                                      'segmentation_inv',
                                                                      'demasking'
                                                                      ])
ds.add_argument('--dataset_dir', default='dataset/')
ds.add_argument('--temples', type=int, nargs='+')
ds.add_argument('--split', type=float, default=0.25)
ds.add_argument('--batch_size', type=int, default=1)
ds.add_argument('--repeat', type=int, default=1)
ds.add_argument('--img_format', default='png')

args = ps.parse_args()
ds_args = [args.temples, args.split, args.batch_size, args.repeat, args.img_format]

if args.dataset_dir[-1] not in ['/', '\\']:
    args.dataset_dir += '/'
if args.log_dir[-1] not in ['/', '\\']:
    args.log_dir += '/'

path = os.path.join(os.getcwd(), args.dataset_dir)
setup_paths(path)

train, validation = get_dataset(args.training_type, ds_args)
#
# model = get_model(args.model, args.training_type)
#
# log_path = os.path.join(os.getcwd(), args.log_dir)
# model.fit(train, args.epochs, path=log_path)
