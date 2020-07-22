import os
import settings
import tensorflow as tf

from datetime import datetime
from tensorflow import keras
from keras_parts import builder
from keras_parts.callbacks import ImageSampling
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
preprocessing.height = settings.IMG_HEIGHT
preprocessing.width = settings.IMG_WIDTH
preprocessing.set_mask()

if settings.TRAINING in ["color_assisted", "de-masking"]:
    repetitions = [1, settings.REPEAT, settings.REPEAT]
elif settings.TRAINING == "masking":
    repetitions = [1, settings.REPEAT, settings.REPEAT, 1]
else:
    repetitions = [1, settings.REPEAT]

if settings.TRAINING == "color_assisted":
    assisted = True
else:
    assisted = False

# --- logs ---
time = datetime.now().strftime('%Y%m%d-%H%M%S')
temples = [str(x) for x in settings.TEMPLES]
temples = ''.join(temples)
resolution = f'{settings.IMG_WIDTH}x{settings.IMG_HEIGHT}'
log_name = f'\\{settings.MODEL}\\{settings.TRAINING}\\'
log_name += f't{temples}-{resolution}-buffer{settings.BUFFER_SIZE}-batch{settings.BATCH_SIZE}-e{settings.EPOCHS}\\{time}'
log_dir = os.path.abspath(settings.LOG_DIR) + log_name


# --- dataset ---
dataset_dir = os.path.abspath(settings.DATASET_DIR)

train, val = data.get_dataset(
    dataset_dir,
    settings.TRAINING,
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
    settings.MODEL, settings.TRAINING, (settings.IMG_HEIGHT, settings.IMG_WIDTH, 3), settings.NORM_TYPE,
)

# --- training ---

# callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
image_sampling = ImageSampling(
    train.take(settings.N_SAMPLES), val.take(settings.N_SAMPLES), settings.FREQUENCY, log_dir=log_dir,
)
checkpoint_dir = 'tmp/checkpoints'
checkpoints = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='val_g_loss',
    mode='min',
    save_best_only=True,
)

model.fit(
    train, epochs=settings.EPOCHS, callbacks=[tensorboard, image_sampling, checkpoints], validation_data=val,
)

if settings.SAVE:
    model_name = '_'.join([resolution, settings.MODEL, settings.TRAINING, temples])
    model.generator.save(f'saved_models/{model_name}')

