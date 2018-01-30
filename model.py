import os

from keras import layers
from keras import models
from keras import optimizers
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


class ModelLoader():

    def __init__(self, n_labels, model_name,
                 saved_weights=None, 
                 optimizer=None, 
                 input_shape=(224, 224, 1)):

        self.n_labels = n_labels
        self.load_model = models.load_model
        self.saved_weights = saved_weights
        self.model_name = model_name
        self.input_shape = input_shape
        self.input_size = input_shape[:2]

        # Loads the specified model
        if self.model_name == 'simple_cnn_binary':
            #print(f'loading {self.model_name}')
            self.model = self.simple_cnn_binary()
            self.loss = 'binary_crossentropy'
            self.optimizer = optimizers.RMSprop(lr=1e-4)
            self.metrics = ['acc']

        elif self.model_name == 'simple_cnn_multi':
            #print('loading {}'.format(self.model_name))
            self.model = self.simple_cnn_multi()
            self.loss = 'categorical_crossentropy'
            self.optimizer = optimizers.RMSprop(lr=1e-4)
            self.metrics = ['acc']

        elif self.model_name == 'simple_cnn_multi_v1':
            #print(f'loading {self.model_name}')
            self.model = self.simple_cnn_multi_v1()
            self.loss = 'categorical_crossentropy'
            self.optimizer = optimizers.Adam(lr=1e-5)
            self.metrics = ['acc']

        elif self.model_name == 'simple_cnn_multi_v2':
            #print(f'loading {self.model_name}')
            self.model = self.simple_cnn_multi_v2()
            self.loss = 'binary_crossentropy'
            self.optimizer = optimizers.RMSprop(lr=1e-4)
            self.metrics = ['acc']

        elif self.model_name == 'vgg16_v0':
            #print(f'loading {self.model_name}')
            self.model = self.small_vgg16_v0()
            self.loss = 'binary_crossentropy'
            self.optimizer = optimizers.RMSprop(lr=1e-5)
            self.metrics = ['acc']

        else:
            raise Exception('No model with name {} found!'.format(model_name))

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

        print(self.model.summary())

    def simple_cnn_binary(self):

        model = models.Sequential()

        #: Conv Network
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Classifier Network - stack of dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def simple_cnn_multi(self):

        model = models.Sequential()
        
        #: Conv Network
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(224, 224, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Classifier Network - stack of dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.n_labels, activation='softmax'))

        return model
    
    def simple_cnn_multi_v1(self):

        model = models.Sequential()
        
        #: Conv Network

        #: Conv Block 1
        model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 1)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Conv Block 2
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        #: Conv Block 3
        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Conv Block 4
        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Classifier Network - stack of dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.n_labels))
        model.add(layers.Activation('softmax'))

        return model

    def simple_cnn_multi_v2(self):

        model = models.Sequential()

        #: Conv Network

        #: Conv Block 1
        model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 1)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Conv Block 2
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Conv Block 3
        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Conv Block 4
        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #: Classifier Network - stack of dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.n_labels))
        model.add(layers.Activation('sigmoid'))

        return model

    def small_vgg16_v0(self):

        conv_base = applications.VGG16(weights="imagenet",
                                       include_top=False,
                                       input_shape=self.input_shape)
        conv_base.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model
        

        
        
