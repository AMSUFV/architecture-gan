import os
import settings

from tensorflow.keras.models import load_model
from utils import data, preprocessing, evaluators


# data.TEST_MODE = True
#
# preprocessing.HEIGHT = settings.IMG_HEIGHT
# preprocessing.WIDTH = settings.IMG_WIDTH
#
# if settings.DATASET in ["color_assisted", "de-masking"]:
#     data.REPETITIONS = [1, settings.REPEAT, settings.REPEAT]
# elif settings.DATASET == "masking":
#     data.REPETITIONS = [1, settings.REPEAT, settings.REPEAT, 1]
# else:
#     data.REPETITIONS = [1, settings.REPEAT]

# dataset_dir = os.path.abspath(settings.DATASET_DIR)
#
# test = data.get_dataset(
#     dataset_dir,
#     settings.DATASET,
#     settings.TEMPLES,
#     settings.SPLIT,
#     settings.BATCH_SIZE,
#     settings.BUFFER_SIZE,
# )
#
# model_dir = os.path.abspath(settings.MODEL_PATH)
# generator = keras.models.load_model(model_dir)
#
# evaluator = L2Evaluator(generator=generator, assisted=True)
# results = evaluator.evaluate(test)

# print(sum(results) / len(results))


def setup():
    data.TEST_MODE = True
    preprocessing.HEIGHT = settings.IMG_HEIGHT
    preprocessing.WIDTH = settings.IMG_WIDTH

    if settings.DATASET in ["color_assisted", "de-masking"]:
        data.REPETITIONS = [1, settings.REPEAT, settings.REPEAT]
    elif settings.DATASET == "masking":
        data.REPETITIONS = [1, settings.REPEAT, settings.REPEAT, 1]
    else:
        data.REPETITIONS = [1, settings.REPEAT]


def evaluate_single_model(data):
    generator = load_model(os.path.abspath(settings.MODEL_PATH))
    evaluator = evaluators.Evaluator(generator=generator, metric=settings.METRIC)
    return evaluator.evaluate(data)


def evaluate_step_model(data):
    segmentator = load_model(os.path.abspath(settings.MODEL_PATH.get('segmenter')))
    color_reconstructor = load_model(os.path.abspath(settings.MODEL_PATH.get('color_reconstructor')))
    reconstructor = load_model(os.path.abspath(settings.MODEL_PATH.get('reconstructor')))
    evaluator = evaluators.StepEvaluator(segmentator, color_reconstructor, reconstructor, metric=settings.METRIC)
    return evaluator.evaluate(data)


def to_file(results: list):
    file = settings.TEST_SAVE_PATH + settings.TEST_FILE_NAME + f'{settings.METRIC.name}.txt'
    with open(file, 'w') as f:
        for result in results:
            f.write(f'{result}\n')


def main():
    setup()

    dataset_dir = os.path.abspath(settings.DATASET_DIR)
    test_dataset = data.get_dataset(
        dataset_dir,
        settings.DATASET,
        settings.TEMPLES,
        settings.SPLIT,
        settings.BATCH_SIZE,
        settings.BUFFER_SIZE,
    )

    if type(settings.MODEL_PATH) is str:
        results = evaluate_single_model(test_dataset)
    else:
        results = evaluate_step_model(test_dataset)

    if settings.TO_FILE:
        to_file(results)


if __name__ == '__main__':
    main()
