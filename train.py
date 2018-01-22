import os
import time

import pandas as pd

from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from model import ModelLoader
from data import get_data_dirs

def save_history(history, fname):

    h_df = pd.DataFrame(history)
    h_df.to_csv("./output/logs/history_{}.csv".format(fname))
    print(history)

    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    return

    # with open(os.path.join("/Users/benja/code/jester/result", 'result_{}.txt'.format(name)), 'w') as fp:
    #     fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
    #     for i in range(nb_epoch):
    #         fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
    #             i, loss[i], acc[i], val_loss[i], val_acc[i]))


def plot_history(hist):

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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

    plt.savefig('train_history_v0.png')
    return

labels_want = ['robot','human']

def train(model_name, dataset_name):

    # Data parameters
    data_dir = get_data_dirs(dataset_name)
    n_images = {'train': 400, 'validation': 50, 'test': 50}
    image_size = (320, 240)

    # Training parameters
    n_epochs = 20
    batch_size = 1
    steps_per_epoch = n_images['train'] // n_epochs // batch_size
    
    # Load data generators
    
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    class_mode = 'binary'

    train_generator = train_datagen.flow_from_directory(
        data_dir['train'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode)

    validation_generator = test_datagen.flow_from_directory(
        data_dir['validation'],
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode)


    #optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay= 1e-6, nesterov=True)
    #Load model

    ml = ModelLoader(n_labels=2, model_name=model_name, image_size=image_size)
    model = ml.model

    #Define callbacks
    checkpointer = ModelCheckpoint(
        filepath='./output/hdf5/{}'.format(model_name) +
        '-{epoch:03d}-{loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    tb = TensorBoard(log_dir='./output/logs')
    early_stopper = EarlyStopping(patience=2)
    csv_logger = CSVLogger('./output/logs/' + model_name + '-' + 'training-' + \
        str(time.time()) + '.log')

    callbacks = [tb, early_stopper, csv_logger, checkpointer]

    # Training
    print('Starting training')

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=n_epochs,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=10,
    )

    model.save('./output/{}.h5'.format(model_name))
    save_history(history, "hist_{}".format(model_name))
    return


if __name__=='__main__':

    model_name = 'small_binary_cnn_v0'
    dataset_name = 'robot_human'

    train(model_name, dataset_name)
