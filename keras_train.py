import os
import settings
import tensorflow as tf

from datetime import datetime
from tensorflow import keras
from keras_parts import builder
from keras_parts.callbacks import ImageSampling
from utils import data
from utils import preprocessing

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

# train, val = data.get_dataset(
#     dataset_dir,
#     settings.TRAINING,
#     settings.TEMPLES,
#     settings.SPLIT,
#     settings.BATCH_SIZE,
#     settings.BUFFER_SIZE,
# )

# for  testing purposes
x = y = tf.random.normal((5, settings.IMG_HEIGHT, settings.IMG_WIDTH, 3))
x = tf.data.Dataset.from_tensor_slices(x).batch(1)
y = tf.data.Dataset.from_tensor_slices(y).batch(1)
train = val = tf.data.Dataset.zip((x, y))

# --- model ---
model = builder.get_model(
    settings.MODEL, settings.TRAINING, (settings.IMG_HEIGHT, settings.IMG_WIDTH, 3)
)

# --- training ---

# callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
image_sampling = ImageSampling(
    train.take(5), val.take(5), settings.FREQUENCY, log_dir=log_dir,
)

model.fit(
    train, epochs=settings.EPOCHS, callbacks=[tensorboard, image_sampling], validation_data=val,
)

# model.generator.save('generator.h5')
