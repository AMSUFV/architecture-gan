import argparse
from utils.dataset_tool import get_dataset, setup_paths

ps = argparse.ArgumentParser()

training = ps.add_argument_group('training')
training.add_argument('--model', default='pix2pix')
training.add_argument('--epochs', type=int, default=100)

ds = ps.add_argument_group('dataset', 'dataset configuration settings')
ds.add_argument('--training_type', default='reconstruction', choices=['color_assisted',
                                                                      'color_reconstruction'
                                                                      'direct',
                                                                      'segmentation',
                                                                      'segmentation_inv'
                                                                      ])
ds.add_argument('--dataset_dir', default='dataset')
ds.add_argument('--temples', type=int, nargs='+')
ds.add_argument('--split', type=float, default=0.25)
ds.add_argument('--batch_size', type=int, default=1)
ds.add_argument('--repeat', type=int, default=1)
ds.add_argument('--img_format', default='png')

args = ps.parse_args()
# dataset = get_dataset(args.option)
ds_args = [args.img_format, args.split, args.batch_size, args.repeat]
setup_paths(args.dataset_dir)
train, validation = get_dataset(args.option, *ds_args)
