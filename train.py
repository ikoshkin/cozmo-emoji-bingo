import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from model import ModelLoader
from data import get_data_dirs, new_folder

def save_history(history, fname):

    h_df = pd.DataFrame(history.history)

    h_df.to_csv("./output/logs/history_{}.csv".format(fname))
    print(history)

    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    return


def plot_history(hist_df, fname=None):

    acc = hist_df['acc']
    val_acc = hist_df['val_acc']
    loss = hist_df['loss']
    val_loss = hist_df['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

    return


labels_want = ['alien', 'devil', 'ghost', 'hearteyes', 'human']
            #    'lipstick', 'octopus', 'poop', 'robot', 'rocket', 'unicorn']

def train(model_name, dataset_name, targets, info):

    #: Data parameters
    data_dir = get_data_dirs(dataset_name)
    n_images = {'train': 300, 'validation': 100, 'test': 100}
    image_size = (320, 240)

    if not os.path.exists("./output/hdf5"):
        new_folder("./output/hdf5")
    if not os.path.exists("./output/logs"):
        new_folder("./output/logs")

    #: Training parameters

    n_epochs = 100
    batch_size = 20
    steps_per_epoch = n_images['train'] * len(targets) // batch_size #// n_epochs
    
    #: Load data generators
    
    # train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    class_mode = 'categorical'

    train_generator = train_datagen.flow_from_directory(
        data_dir['train'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        # save_to_dir='./images/keras_train'
        )

    validation_generator = test_datagen.flow_from_directory(
        data_dir['validation'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode)


    #: Load model
    n_labels = len(targets)
    ml = ModelLoader(n_labels=n_labels, model_name=model_name,
                     image_size=image_size)
    model = ml.model

    #: Define callbacks
    checkpointer = ModelCheckpoint(
        filepath='./output/hdf5/{}'.format(model_name) +
        '-{epoch:03d}-{loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True, save_weights_only=True)

    callbacks = [checkpointer]

    #: Training
    print('Starting training')

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=20
    )

    model.save('./output/{}_{}.h5'.format(model_name, info))
    save_history(history, "hist_{}_{}".format(model_name, info))

    return history


if __name__=='__main__':

    # model_name = 'vgg16_v0'
    # dataset_name = 'robot_human'
    

    model_name = 'simple_cnn_multi'
    dataset_name = 'five_multiclass'
    model_run_info = '5c_with_aug'

    targets = ['alien', 'devil', 'ghost', 'hearteyes', 'human']#,
            #    'lipstick', 'octopus', 'poop', 'robot', 'rocket', 'unicorn']

    history=train(model_name, dataset_name, targets, model_run_info)
    h_df = pd.DataFrame(history.history)
    h_df.to_csv('five_multiclass.csv')

    # plot_history(h_df, fname='cnn_multi_h_r.png')
