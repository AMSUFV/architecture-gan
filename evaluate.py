import os
import settings

from tensorflow.keras.models import load_model
from utils import data, preprocessing, evaluators


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


def evaluate_single_model(dataset):
    generator = load_model(os.path.abspath(settings.MODEL_PATH))
    evaluator = evaluators.Evaluator(generator=generator, metric=settings.METRIC)
    return evaluator.evaluate(dataset)


def evaluate_step_model(dataset):
    segmentator = load_model(os.path.abspath(settings.MODEL_PATH.get('segmenter')))
    color_reconstructor = load_model(os.path.abspath(settings.MODEL_PATH.get('color_reconstructor')))
    reconstructor = load_model(os.path.abspath(settings.MODEL_PATH.get('reconstructor')))
    evaluator = evaluators.StepEvaluator(segmentator, color_reconstructor, reconstructor, metric=settings.METRIC)
    return evaluator.evaluate(dataset)


def get_data():
    dataset_dir = os.path.abspath(settings.DATASET_DIR)
    return data.get_dataset(
        dataset_dir,
        settings.DATASET,
        settings.TEMPLES,
        settings.SPLIT,
        settings.BATCH_SIZE,
        settings.BUFFER_SIZE,
    )


def get_evaluator():
    if type(settings.MODEL_PATH) is str:
        generator = load_model(os.path.abspath(settings.MODEL_PATH))
        evaluator = evaluators.Evaluator(generator=generator, metric=settings.METRIC)
    else:
        segmentator = load_model(os.path.abspath(settings.MODEL_PATH.get('segmenter')))
        color_reconstructor = load_model(os.path.abspath(settings.MODEL_PATH.get('color_reconstructor')))
        reconstructor = load_model(os.path.abspath(settings.MODEL_PATH.get('reconstructor')))
        evaluator = evaluators.StepEvaluator(segmentator, color_reconstructor, reconstructor, metric=settings.METRIC)
    return evaluator


def to_file(results: list):
    file = settings.TEST_SAVE_PATH + settings.TEST_FILE_NAME + f'{settings.METRIC.name}.txt'
    with open(file, 'w') as f:
        for result in results:
            f.write(f'{result}'.strip('[]') + '\n')


def main():
    setup()
    dataset = get_data()
    evaluator = get_evaluator()
    results = evaluator.evaluate(dataset)
    if settings.TO_FILE:
        to_file(results)


if __name__ == '__main__':
    main()
