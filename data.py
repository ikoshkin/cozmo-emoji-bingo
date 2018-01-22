import os
import shutil
import random
from keras.preprocessing.image import ImageDataGenerator

DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(DIR, 'images')

IMG_W, IMG_H, IMG_D = 320, 240, 3


def get_data_dirs(dataset_name):

    DATASET_DIR = os.path.join(IMAGES_DIR, dataset_name)
    assert os.path.exists(DATASET_DIR) is True

    train_dir = os.path.join(DATASET_DIR, 'train')
    validation_dir = os.path.join(DATASET_DIR, 'validation')
    test_dir = os.path.join(DATASET_DIR, 'test')

    return (train_dir, validation_dir, test_dir)


def new_folder(path):
    os.makedirs(path, exist_ok=True)
    return


def build_dataset(labels=None, n_train=400, n_val=50, n_test=50, seed=42):

    dataset_name = '_'.join(labels)

    original_dataset_dir = os.path.join(IMAGES_DIR, 'originals')
    base_dir = os.path.join(IMAGES_DIR, dataset_name)
    new_folder(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    new_folder(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    new_folder(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    new_folder(test_dir)

    for l in labels:
        original_l_dir = os.path.join(original_dataset_dir, l)

        train_l_dir = os.path.join(train_dir, l)
        new_folder(train_l_dir)

        validation_l_dir = os.path.join(validation_dir, l)
        new_folder(validation_l_dir)

        test_l_dir = os.path.join(test_dir, l)
        new_folder(test_l_dir)

        fnames = [f for f in os.listdir(original_l_dir) if f.endswith('.jpeg')]
        random.seed(42)
        random.shuffle(fnames)

        for fname in fnames[:n_train]:
            src = os.path.join(original_l_dir, fname)
            dst = os.path.join(train_l_dir, fname)
            shutil.copyfile(src, dst)

        for fname in fnames[n_train:(n_train + n_val)]:
            src = os.path.join(original_l_dir, fname)
            dst = os.path.join(validation_l_dir, fname)
            shutil.copyfile(src, dst)

        for fname in fnames[(n_train + n_val):n_test]:
            src = os.path.join(original_l_dir, fname)
            dst = os.path.join(test_l_dir, fname)
            shutil.copyfile(src, dst)
    return


if __name__ == '__main__':
    # print(get_dirs('doll_seashell'))

    labels = ['robot', 'human']
    build_dataset(labels)
