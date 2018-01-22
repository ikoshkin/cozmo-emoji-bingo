import os

import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


class ModelLoader():

    def __init__(self, n_labels, model_name,
                 saved_weights=None, optimizer=None, image_size=(320, 240)):

        self.n_labels = n_labels
        self.load_model = models.load_model
        self.saved_weights = saved_weights
        self.model_name = model_name

        # Loads the specified model
        if self.model_name == 'small_binary_cnn_v0':
            print('Loading Small Binary CNN v0')
            self.input_shape = (image_size + (3,))
            self.model = self.small_cnn_v0()
            self.loss = 'binary_crossentropy'
            self.optimizer = optimizers.RMSprop(lr=1e-4)
            self.metrics = ['acc']

        elif self.model_name == 'small_binary_cnn_v1':
            print('Loading Small Binary CNN v0')
            self.input_shape = (image_size + (3,))
            self.model = self.small_cnn_v0()
            self.loss = 'binary_crossentropy'
            self.optimizer = optimizers.RMSprop(lr=1e-4)
            self.metrics = ['acc']

        else:
            raise Exception('No model with name {} found!'.format(model_name))

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

        print(self.model.summary())

    def small_cnn_v0(self):

        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def small_cnn_v1(self):
        pass
