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

    return {'train': train_dir, 'validation': validation_dir, 'test': test_dir}


def new_folder(path):
    os.makedirs(path, exist_ok=True)
    return


def build_dataset(targets, dataset_name, n_images, seed=42):

    if dataset_name is None:
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

    for target in targets:
        original_target_dir = os.path.join(original_dataset_dir, target)

        train_target_dir = os.path.join(train_dir, target)
        new_folder(train_target_dir)

        validation_target_dir = os.path.join(validation_dir, target)
        new_folder(validation_target_dir)

        test_target_dir = os.path.join(test_dir, target)
        new_folder(test_target_dir)

        fnames = [f for f in os.listdir(original_target_dir) if f.endswith('.jpeg')]
        random.seed(42)
        random.shuffle(fnames)

        idx_train = n_images['train']
        idx_val = idx_train + n_images['validation']
        idx_test = idx_val + n_images['test']

        for fname in fnames[:idx_train]:
            src = os.path.join(original_target_dir, fname)
            dst = os.path.join(train_target_dir, fname)
            shutil.copyfile(src, dst)

        for fname in fnames[idx_train:idx_val]:
            src = os.path.join(original_target_dir, fname)
            dst = os.path.join(validation_target_dir, fname)
            shutil.copyfile(src, dst)

        for fname in fnames[idx_val:idx_test]:
            src = os.path.join(original_target_dir, fname)
            dst = os.path.join(test_target_dir, fname)
            shutil.copyfile(src, dst)

        print('total training {} images: {}'.format(
            target, len(os.listdir(train_target_dir))))
        print('total validating {} images: {}'.format(
            target, len(os.listdir(validation_target_dir))))
        print('total testing {} images: {}'.format(
            target, len(os.listdir(test_target_dir))))

    return


if __name__ == '__main__':

    targets = ['human', 'robot']
    n_images = {'train': 400, 'validation': 50, 'test': 50}
    dataset_name = 'human_robot_multiclass'

    build_dataset(targets, dataset_name, n_images)
