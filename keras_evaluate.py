import os
import settings

from tensorflow import keras
from keras_models.evaluators import L2Evaluator
from utils import data, preprocessing

data.TEST_MODE = True


preprocessing.HEIGHT = settings.IMG_HEIGHT
preprocessing.WIDTH = settings.IMG_WIDTH

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

dataset_dir = os.path.abspath(settings.DATASET_DIR)

test = data.get_dataset(
    dataset_dir,
    settings.DATASET,
    settings.TEMPLES,
    settings.SPLIT,
    settings.BATCH_SIZE,
    settings.BUFFER_SIZE,
)

model_dir = os.path.abspath(settings.MODEL_PATH)
generator = keras.models.load_model(model_dir)

evaluator = L2Evaluator(generator=generator, assisted=True)
results = evaluator.evaluate(test)

print(sum(results)/len(results))

if settings.TO_FILE:
    file = settings.TEST_SAVE_PATH + settings.TEST_FILE_NAME
    with open(file, 'w') as f:
        for result in results:
            f.write(f'{result}\n')
