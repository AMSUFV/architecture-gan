import argparse
from utils import dataset_tool
from models.pix2pix import Pix2Pix
from models.hybrid_reconstuctor import HybridReconstructor

parser = argparse.ArgumentParser(description='Launch model training')

parser.add_argument('-temples', '-t', nargs='*', type=int, required=True,
                    help='Temples to use for training, numeric separated by spaces.')

parser.add_argument('-p', default='./dataset', type=str,
                    help='Dataset path.')

parser.add_argument('-mode', default='reconstruction', required=True,
                    help='Training mode. Either segmentation, desegmentation, reconstruction_real,'
                         'reconstruction_color,or hybrid')

parser.add_argument('-model', '-m', type=str, required=True,
                    help='Selects a model to train; either pix2pix or hybrid.')

parser.add_argument('-epochs', '-e', default=50, type=int,
                    help='Number of epochs to train the model for.')

parser.add_argument('-split', default=0.25, type=float,
                    help='Train/validation split.')

parser.add_argument('-repeat', '-r', default=2, type=int, required=True,
                    help='Ruins to temple ratio.')

parser.add_argument('-logdir', default=None, type=str,
                    help='Path to Tensorboard log directory')

parser.add_argument('-save', default=True, type=bool,
                    help='Wether or not to save the model after training.')

args = parser.parse_args()

temple_list = [f'temple_{x}' for x in args.temples]
dataset_tool.setup_paths(args.p)

if args.model == 'pix2pix':
    model = Pix2Pix(log_dir=args.logdir, autobuild=True)
elif args.model == 'hybrid':
    model = HybridReconstructor(log_dir=args.logdir, autobuild=False)
    model.build_generator(heads=2, inplace=True)
    model.build_discriminator(inplace=True)
else:
    raise Exception('Unsupported model.')

if args.mode == 'segmentation':
    train, validation = dataset_tool.get_dataset_segmentation(temple_list, split=args.split, repeat=args.repeat)
elif args.mode == 'desegmentation':
    train, validation = dataset_tool.get_dataset_segmentation(temple_list, split=args.split, repeat=args.repeat,
                                                              inverse=True)
elif args.mode == 'hybrid':
    train, validation = dataset_tool.get_dataset_dual_input(temple_list, split=args.split, repeat=args.repeat)
elif 'color' in args.mode:
    train, validation = dataset_tool.get_dataset_reconstruction(temple_list, split=args.split, mode='color',
                                                                repeat=args.repeat)
elif 'real' in args.mode:
    train, validation = dataset_tool.get_dataset_reconstruction(temple_list, split=args.split, mode='real',
                                                                repeat=args.repeat)
else:
    raise Exception('Unsupported method. See -h for supported methods.')

model.fit(train, validation, args.epochs)

# naming
model_name = f'{args.mode}_{args.model}'
model.generator.save(f'./trained_models/{model_name}.h5')
