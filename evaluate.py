import os
import settings
import tensorflow as tf

from models.pix2pix import StepModel
from parts import losses
from statistics import mean
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model
from utils import data, evaluators, preprocessing
from utils.metrics import Metric


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


def get_c2st():
    if type(settings.MODEL_PATH) is str:
        generator = load_model(os.path.abspath(settings.MODEL_PATH))
    else:
        generator = StepModel(
            color_reconstructor=settings.MODEL_PATH.get('color_reconstructor'),
            segmenter=settings.MODEL_PATH.get('segmenter'),
            reconstructor=settings.MODEL_PATH.get('reconstructor')
        )
    optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    loss = losses.Pix2PixLosses.loss_d
    c2st = evaluators.C2ST(generator)
    c2st.compile(d_optimizer=optimizer, d_loss_fn=loss)
    return c2st


def evaluate_discriminator(discriminator, test_dataset):
    bce = BinaryAccuracy(threshold=0.2)
    file = settings.TEST_SAVE_PATH + settings.TEST_FILE_NAME + 'C2ST.txt'

    results = []
    with open(file, 'a+') as f:
        for *x, y in test_dataset:
            if type(x) == list:
                x = x[0]
            dx = discriminator([x, y], training=False)
            result = bce(tf.zeros_like(dx), dx).numpy()
            results.append(result)
            # f.write(f'{result}\n')
        f.write(f'Mean: {mean(results)}\n')


def to_file(results: list):
    file = settings.TEST_SAVE_PATH + settings.TEST_FILE_NAME + f'{settings.METRIC.name}.txt'
    with open(file, 'w') as f:
        for result in results:
            f.write(f'{result}'.strip('[]') + '\n')


def main():
    setup()

    if settings.METRIC is Metric:
        dataset = get_data()
        evaluator = get_evaluator()
        results = evaluator.evaluate(dataset)
        if settings.TO_FILE:
            to_file(results)

    elif settings.METRIC == 'C2ST':
        for _ in range(4):
            data.TEST_MODE = False
            train, validation = get_data()
            c2st = get_c2st()
            c2st.fit(train, epochs=settings.EPOCHS)
            c2st = c2st.discriminator
            evaluate_discriminator(c2st, validation)


if __name__ == '__main__':
    main()
