from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K

import pandas as pd
import datetime
import os
import shutil

def data():
    nb_classes = 10
    # the data, shuffled and split between train and test sets
    
    n_images = {'train': 300, 'validation': 100, 'test': 100}




    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255

    '''
    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    '''

    train_datagen = ImageDataGenerator(
        rescale=1 / 255.,
        rotation_range=15,
        width_shift_range=0.3,
        height_shift_range=0.3,
        featurewise_std_normalization=True,
        shear_range=0.3,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant',
        cval=255. / 2)

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    # datagen.fit(X_train)

    return train_datagen, val_datagen, test_datagen  # , X_train, Y_train, X_test, Y_test


def model(train_datagen, val_datagen, test_datagen):

    # input image dimensions
    img_rows, img_cols = 244, 244
    image_size = (img_rows, img_cols)

    # the CIFAR10 images are RGB
    img_channels = 1
    input_shape = (image_size + (img_channels,))

    color_mode = 'grayscale'





    model = Sequential()

    model.add(Convolution2D(32, 3, 3,
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout({{uniform(0, 1)}}))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))





    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])






    train_dir = './images/ten_multiclass/train'
    val_dir = './images/ten_multiclass/validation'
    test_dir = './images/ten_multiclass/test'

    class_mode = 'categorical'

    n_epochs = 1
    batch_size = 16
    steps_per_epoch = 300 * 10 // batch_size
    # steps_per_epoch = 300

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode=class_mode)

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode=class_mode)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode=class_mode)


    ''' 
    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
    '''

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=20
    )

    drop_history(history.history)

    score, acc = model.evaluate_generator(test_generator, verbose=1, steps=20)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def drop_history(h):

    runname = 'multiten'


    h_df = pd.DataFrame(h)
    hdfname = './hyperas/history{rname}_{timestamp}.csv'.format(rname=runname,
        timestamp=datetime.datetime.now().strftime('%m%d_%H%M%S'))
    h_df.to_csv(hdfname)


if __name__ == '__main__':


    if os.path.exists('./hyperas'):
        shutil.rmtree('./hyperas')
    if not os.path.exists('./hyperas'):
        os.makedirs('./hyperas')

    

    train_datagen, val_datagen, test_datagen = data()

    test_dir = './images/ten_multiclass/test'

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(244, 244),
        color_mode='grayscale',
        batch_size=16,
        class_mode='categorical')

    functions = [drop_history]

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          functions=functions,
                                          max_evals=5,
                                          trials=Trials(),
                                          eval_space=True)

    runname = 'multiten'
    h5name = './hyperas/history{rname}_{timestamp}.csv'.format(rname=runname,
                                                            timestamp=datetime.datetime.now().strftime('%m%d_%H%M%S'))
    best_model.save('./hyperas/{}_{}.h5'.format(model_name, info))

    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(test_generator, steps=20)) 
    
    if K.backend() == 'tensorflow':
            K.clear_session()
