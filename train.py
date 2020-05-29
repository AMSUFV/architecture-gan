import argparse
import os

from datetime import datetime

from utils import get_model
from utils import preprocessing
from utils import data

ps = argparse.ArgumentParser()

training = ps.add_argument_group('training')
training.add_argument('--epochs', type=int, default=100)
training.add_argument('--log_dir', default='logs/')
training.add_argument('--log_images', default=True)
training.add_argument('--frequency', default=1)

mod = ps.add_argument_group('model', 'model configuration settings')
mod.add_argument('--model', default='pix2pix')
# mod.add_argument('--heads', type=int, default=1)
mod.add_argument('--dim', type=int, default=64)
mod.add_argument('--down_blocks', type=int, default=8)
mod.add_argument('--downsamplings', type=int, default=4)
mod.add_argument('--norm_type', default='batchnorm')

ds = ps.add_argument_group('dataset', 'dataset configuration settings')
ds.add_argument('--training', default='reconstruction', choices=['color_assisted',
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
ds.add_argument('--img_height', type=int, default=512)
ds.add_argument('--img_width', type=int, default=512)

args = ps.parse_args()

# making sure the image will make it through the bottleneck
data.validate(args.model, args.img_width, args.img_height, args.down_blocks)

# preprocessing setup
preprocessing.height = args.img_height
preprocessing.width = args.img_width
img_format = args.img_format.strip('.').lower()
preprocessing.setup(img_format)

# data repetition check
if args.training in ['color_assisted', 'de-masking']:
    data.repetitions = [1, args.repeat, args.repeat]
elif args.training == 'masking':
    data.repetitions = [1, args.repeat, args.repeat, 1]
else:
    data.repetitions = [1, args.repeat]

# absolut path check
if not os.path.isabs(args.log_dir):
    args.log_dir = os.path.abspath(args.log_dir)

# head check
if args.training.lower() == 'color_assisted':
    heads = 2
else:
    heads = 1

# dataset
ds_args = [args.temples, args.split, args.batch_size, args.buffer_size]
train, val = data.get_dataset(args.dataset_dir, args.training, *ds_args)

# model
model_args = [(None, None, 3), args.norm_type, heads]
model = get_model(args.model, args.training, *model_args)
model.summary()

# logs
time = datetime.now().strftime('%Y%m%d-%H%M%S')
temples = [str(x) for x in args.temples]
temples = ''.join(temples)
resolution = f'{args.img_width}x{args.img_height}'
model_dir = f'/{args.model}/{args.training}/'
model_dir += f't{temples}-{resolution}-buffer{args.buffer_size}-batch{args.batch_size}/{time}'
log_path = os.path.join(os.getcwd(), args.log_dir + model_dir)

# training
model.fit(train, args.epochs, path=log_path, log_images=args.log_images, frequency=args.frequency)

# saving
model_name = '.'.join([resolution, args.model, args.training, f't{temples}'])
model.save(f'{model_name}.h5')
