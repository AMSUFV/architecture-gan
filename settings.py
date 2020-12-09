"""Settings for the temple reconstruction training.

Training parameters:

- MODEL: Model to use. Available:
    * pix2pix
    * resnet
- NORM_TYPE: Type of normalization to apply in the building blocks. Avaliable:
    * batchnorm: Batch normalization
    * instancenorm: Instance normalization
- LOG_DIR: Base folder to keep the training logs at
- LOG_IMAGES: Whether or not to log images to tensorboard during training
- FREQUENCY: Frequency of image logging


Dataset parameters:

- DATASET: Type of dataset. Available:
    * color_assisted:       ((temple ruins, segmented temple), temple)
    * color_reconstruction: (temple ruins segmented, temple segmented)
    * reconstruction:       (temple ruins, temple)
    * segmentation:         (temple segmented, temple)
    * de-segmentation:      (temple, temple segmented)
    * masking:              (temple ruins, temple ruins masked); aimed at marking the missing areas
    * de-masking:           (temple ruins masked, temple); aimed at reconstructing the marked areas
    * text_assisted         ((temple ruins, temple description), temple)
- DATASET_DIR: Base folder of the dataset
- TEMPLES: Temples to use during training, e.g. temple_0, temple_1, etc.
- SPLIT: Train/validation split
- BATCH_SIZE: Batch size
- BUFFER_SIZE: Size of the shuffling buffer. This should be set higher than the images per temple. Ideally, it should
  be set to the size of the dataset
- REPEAT: Number of ruins models per temple model used during training
- IMG_HEIGHT: Image height
- IMG_WIDTH: Image width

"""
from utils import metrics

# training params
MODEL = 'pix2pix'
NORM_TYPE = 'batch'
EPOCHS = 5
LOG_DIR = 'logs/'
LOG_IMAGES = False
N_SAMPLES = 4
FREQUENCY = 3
SAVE = False
SAVE_PATH = 'saved_models/'
RESTORE = False

# dataset params
DATASET = 'reconstruction'
DATASET_DIR = 'dataset/'
TEMPLES: list = [0, 9]
SPLIT = 0.2
BATCH_SIZE = 1
BUFFER_SIZE = 1200
REPEAT = 2
IMG_HEIGHT = 256
IMG_WIDTH = 512

# GPU limits
GPU_LIMIT = None

# test params
# MODEL_PATH = 'saved_models/512x256_pix2pix_batch_color_assisted_12345678'
MODEL_PATH = dict(
    segmenter='saved_models/512x256_pix2pix_batch_segmentation_12345678',
    color_reconstructor='saved_models/512x256_pix2pix_batch_color_reconstruction_12345678',
    reconstructor='saved_models/512x256_pix2pix_batch_color_assisted_12345678'
)
TO_FILE = True
TEST_SAVE_PATH = 'evaluation_results/'
TEST_FILE_NAME = 'step_model/'
METRIC = 'C2ST'
