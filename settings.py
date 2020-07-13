"""Settings for the temple reconstruction training.

Training parameters:

MODEL: Model to use. Available:
    * pix2pix
    * resnet
NORM_TYPE: Type of normalization to apply in the building blocks. Avaliable:
    * batchnorm: Batch normalization
    * instancenorm: Instance normalization
TRAINING: Type of training. Available:
    * color_assisted:       ((temple ruins, segmented temple), temple)
    * color_reconstruction: (temple ruins segmented, temple segmented)
    * reconstruction:       (temple ruins, temple)
    * segmentation:         (temple segmented, temple)
    * de-segmentation:      (temple, temple segmented)
    * masking:              (temple ruins, temple ruins masked); aimed at marking the missing areas
    * de-masking            (temple ruins masked, temple); aimed at reconstructing the marked areas
    * text_assisted         ((temple ruins, temple description), temple)
LOG_DIR: Base folder to keep the training logs at
LOG_IMAGES: Whether or not to log images to tensorboard during training
FREQUENCY: Frequency of image logging


Dataset parameters:

DATASET_DIR: Base folder of the dataset
TEMPLES: Temples to use during training, e.g. temple_0, temple_1, etc.
SPLIT: Train/validation split
BATCH_SIZE: Batch size
BUFFER_SIZE: Size of the shuffling buffer. This should be set higher than the images per temple. Ideally, it should
be set to the size of the dataset
REPEAT: Number of ruins models per temple model used during training
IMG_HEIGHT: Image height
IMG_WIDTH: Image width

"""

# training params
MODEL = 'pix2pix'
NORM_TYPE = 'batch'
TRAINING = 'reconstruction'
EPOCHS = 5
LOG_DIR = 'logs/'
LOG_IMAGES = True
N_SAMPLES = 4
FREQUENCY = 1
SAVE = False

# dataset params
DATASET_DIR = 'dataset/'
TEMPLES = [0]
SPLIT = 0.2
BATCH_SIZE = 1
BUFFER_SIZE = 400
REPEAT = 1
IMG_HEIGHT = 256
IMG_WIDTH = 512

