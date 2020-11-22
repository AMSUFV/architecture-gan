import os
import settings
import tensorflow as tf

from datetime import datetime
from tensorflow import keras
from parts import builder
from parts.callbacks import ImageSampling
from utils import data
from utils import preprocessing

# -- gpu memory limit --
if settings.GPU_LIMIT:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=settings.GPU_LIMIT)]
        )
    except RuntimeError as e:
        print(e)

# --- setup ---
preprocessing.HEIGHT = settings.IMG_HEIGHT
preprocessing.WIDTH = settings.IMG_WIDTH
preprocessing.MASK = preprocessing.get_mask()

if settings.DATASET in ["color_assisted", "de-masking"]:
    data.REPETITIONS = [1, settings.REPEAT, settings.REPEAT]
elif settings.DATASET == "masking":
    data.REPETITIONS = [1, settings.REPEAT, settings.REPEAT, 1]
else:
    data.REPETITIONS = [1, settings.REPEAT]

if settings.DATASET == "color_assisted":
    assisted = True
else:
    assisted = False

# --- logs ---
time = datetime.now().strftime('%Y%m%d-%H%M%S')
temples = [str(x) for x in settings.TEMPLES]
temples = ''.join(temples)
resolution = f'{settings.IMG_WIDTH}x{settings.IMG_HEIGHT}'
log_name = f'\\{settings.MODEL}\\{settings.DATASET}\\'
log_name += f'{settings.NORM_TYPE}_norm\\t{temples}-{resolution}-buffer{settings.BUFFER_SIZE}-' + \
            f'batch{settings.BATCH_SIZE}-e{settings.EPOCHS}\\{time}'
log_dir = os.path.abspath(settings.LOG_DIR) + log_name

# --- dataset ---
dataset_dir = os.path.abspath(settings.DATASET_DIR)

train, val = data.get_dataset(
    dataset_dir,
    settings.DATASET,
    settings.TEMPLES,
    settings.SPLIT,
    settings.BATCH_SIZE,
    settings.BUFFER_SIZE,
)

# for  testing purposes
# x = y = tf.random.normal((5, settings.IMG_HEIGHT, settings.IMG_WIDTH, 3))
# x = tf.data.Dataset.from_tensor_slices(x).batch(1)
# y = tf.data.Dataset.from_tensor_slices(y).batch(1)
# train = val = tf.data.Dataset.zip((x, y))

# --- model ---
model = builder.get_model(
    settings.MODEL,
    settings.DATASET,
    (settings.IMG_HEIGHT, settings.IMG_WIDTH, 3),
    settings.NORM_TYPE,
)

# --- training ---

# callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
image_sampling = ImageSampling(
    train.take(settings.N_SAMPLES),
    val.take(settings.N_SAMPLES),
    settings.FREQUENCY,
    log_dir=log_dir + '\\images',
    assisted=assisted,
)
checkpoint_dir = 'tmp/checkpoints'
checkpoints = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='val_g_loss',
    mode='min',
    save_best_only=True,
)

if settings.RESTORE:
    model.load_weights(checkpoint_dir)

model.fit(
    train,
    epochs=settings.EPOCHS,
    callbacks=[tensorboard, image_sampling, checkpoints],
    validation_data=val,
)

if settings.SAVE:
    model_name = '_'.join([
        resolution,
        settings.MODEL,
        settings.NORM_TYPE,
        settings.DATASET,
        temples
    ])
    model.generator.save(f'{settings.SAVE_PATH}{model_name}')
