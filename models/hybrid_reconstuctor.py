import glob
import tensorflow as tf
import numpy as np
from models.pix2pix import Pix2Pix
from models.pix2pix import downsample, upsample
from utils import custom_preprocessing as cp


class HybridReconstuctor(Pix2Pix):
    def get_dataset(self, temples, split=0.2, dataset_path=None, ruins_per_temple=2):
        if dataset_path is None:
            dataset_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\\'

        datasets = []
        for i, temple in enumerate(temples):

            ruins_path = dataset_path + r'\temples_ruins\\' + temple + '*'
            colors_path = dataset_path + r'\colors_temples\colors_' + temple
            temple_path = dataset_path + r'\temples\\' + temple

            datasets.append(self.get_single_dataset(ruins_path, temple_path, colors_path))

        train_dataset = datasets[0]
        datasets.pop(0)
        for dataset in datasets:
            train_dataset = train_dataset.concatenate(dataset)
        buffer_size = len(temples) * ruins_per_temple * 300
        train_dataset = train_dataset.shuffle(buffer_size)

        return train_dataset

    @staticmethod
    def get_single_dataset(ruins_path, temple_path, colors_path, split=0.2, mode='train'):
        if mode == 'train':
            preprocessing_function = cp.load_images_train
        else:
            preprocessing_function = cp.load_images_test

        ruins_path_list = glob.glob(ruins_path + r'\*.png')
        colors_path_list = glob.glob(colors_path + r'\*.png')
        temple_path_list = glob.glob(temple_path + r'\*.png')

        batch_size = 1
        # buffer_size = len(ruins_path_list)

        repetition = len(ruins_path_list) // len(temple_path_list)

        ruins_dataset = tf.data.Dataset.list_files(ruins_path_list, shuffle=False)
        colors_dataset = tf.data.Dataset.list_files(colors_path_list, shuffle=False)
        temple_dataset = tf.data.Dataset.list_files(temple_path_list, shuffle=False)

        colors_dataset = colors_dataset.repeat(repetition)
        temple_dataset = temple_dataset.repeat(repetition)

        train_dataset = tf.data.Dataset.zip((ruins_dataset, temple_dataset, colors_dataset))
        train_dataset = train_dataset.map(preprocessing_function).batch(batch_size)

        return train_dataset

    @staticmethod
    def build_generator():
        down_stack = [
            downsample(64, 4, apply_batchnorm=False),
            downsample(128, 4),
            downsample(256, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4),
            upsample(256, 4),
            upsample(128, 4),
            upsample(64, 4),
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                               kernel_initializer=initializer, activation='tanh')

        concat = tf.keras.layers.Concatenate()

        ruin_image = tf.keras.layers.Input(shape=[None, None, 3], name='ruin_image')
        color_image = tf.keras.layers.Input(shape=[None, None, 3], name='color_image')
        x = tf.keras.layers.concatenate([ruin_image, color_image])

        # downsampling
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # upsampling and connecting
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=[ruin_image, color_image], outputs=x)

    @tf.function
    def train_step(self, ruin, temple, color):
        with tf.GradientTape(persistent=True) as tape:
            gen_output = self.generator([ruin, color], training=True)

            disc_real = self.discriminator([ruin, temple], training=True)
            disc_generated = self.discriminator([ruin, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated, gen_output, temple)
            disc_loss = self.discriminator_loss(disc_real, disc_generated)

        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def fit(self, train_ds, test_ds=None, epochs=100):
        for epoch in range(epochs):
            # Train
            for ruin, temple, color in train_ds:
                self.train_step(ruin, temple, color)

            self._train_predict(train_ds, self.train_summary_writer, 'train', epoch)

    # prediction methods
    def predict(self, dataset, log_path, samples):
        writer = self._set_logdir(log_path, 'predict')
        step = 0
        if samples == 'all':
            target = dataset
        else:
            target = dataset.take(samples)

        for x, y, z in target:
            # x is the input
            prediction = self.generator([x, z], training=False)
            stack = tf.stack([x, prediction, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)

            with writer.as_default():
                tf.summary.image('predictions', stack, step=step, max_outputs=3)

            step += 1

    def _train_predict(self, dataset, writer, name, step):
        for x, y, z in dataset.take(1):
            generated = self.generator([x, z], training=False)
            stack = tf.stack([x, generated, y], axis=0) * 0.5 + 0.5
            stack = tf.squeeze(stack)
            with writer.as_default():
                tf.summary.image(name, stack, step=step, max_outputs=3)


def main():
    log_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\logs\ruins_and_segmentations'
    ds_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset'
    cp.RESIZE_FACTOR = 1.3
    reconstructor = HybridReconstuctor(log_dir=log_path)
    train = reconstructor.get_dataset(['temple_0', 'temple_1', 'temple_5'], dataset_path=ds_path)
    reconstructor.fit(train, 50)


def predict_batch(target='temple_6', ruins=1):
    temple = target

    log_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\logs\ruins_and_segmentations\\' + temple

    ds_path = r'C:\Users\Ceiec06\Documents\GitHub\ARQGAN\dataset\\'
    ruins = ds_path + r'temples_ruins\\' + temple + f'_ruins_{ruins}'
    colors = ds_path + r'colors_temples\colors_' + temple
    temples = ds_path + r'temples\\' + temple

    reconstructor = HybridReconstuctor(gen_path='../trained_models/ruinseg.h5')
    predict_ds = reconstructor.get_single_dataset(ruins, temples, colors, mode='predict')
    reconstructor.predict(predict_ds, log_path, samples='all')


if __name__ == '__main__':
    predict_batch(target='temple_6', ruins=1)
