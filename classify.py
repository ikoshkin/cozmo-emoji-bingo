import os
import sys
import random

import numpy as np
import pandas as pd
import h5py

from keras import models
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


class_names = {
    0: 'alien',
    1: 'devil',
    2: 'ghost',
    3: 'hearteyes',
    4: 'human',
    5: 'lipstick',
    6: 'octopus',
    7: 'poop',
    8: 'robot',
    9: 'rocket',
    10: 'unicorn'
}

h5_file = './output-aws/all_bin_sigm-130pm/output/hdf5/simple_cnn_multi_v2_all_bin_sigm.h5'
hdf5_file = './output-aws/all_bin_sigm-130pm/output/hdf5/simple_cnn_multi_v2.hdf5'

class ImageClassifier():
    
    def __init__(self, m_path=h5_file, w_path=hdf5_file):

        self.class_names = class_names
        self.top_n = None
        self.img_file = None
        # self.img_size = (320, 240)
        self.img_size = (224, 224)
        self.grayscale = True
        self.model = self.load_model_and_weights(m_path, w_path)

    def load_model_and_weights(self, m_path, w_path):
        model = models.load_model(m_path)
        model.load_weights(w_path)
        return model

    def classify(self, img_file):

        img = load_img(img_file,
                       target_size=self.img_size,
                       grayscale=self.grayscale)

        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.

        pred_probs = self.model.predict(x)

        n = 3

        top_n_preds = pred_probs.argsort(axis=1)[0][::-1][:n]
        results = [(class_names.get(i), pred_probs[0][i]) for i in top_n_preds]

        self.img_file = img_file
        self.top_n = results

        return results[0]

    def plot_results(self, save_to=None):
        if self.top_n is None:
            print('clssify first: self.classify(imgpath)')
        else:
            results = self.top_n

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        ax.imshow(mpimg.imread(self.img_file))

        for i, r in enumerate(results):
            c_name, c_prob = r
            xy = (10, i * 20 + 20)
            ax.annotate('{0}: {1:0.4f}'.format(c_name, c_prob),
                        fontsize=12, xy=xy, color='cyan')

        if save_to is not None:
            plt.savefig(save_to)

    def print_top_n(self):
        if self.top_n is not None:
            print('====================================')
            for i, r in enumerate(self.top_n):
                print('{0}) emoji: {1:12} prob: {2:0.3f}'.format(i, r[0], r[1]))
    
def pick_random_image():
    global class_names
    label = class_names.get(random.randint(0, 9))
    path = os.path.join('./images/originals', label)
    fname = random.choice(os.listdir(path))
    return os.path.join(path, fname)


if __name__ == '__main__':

    some_emoji = pick_random_image()
    emoji_clf = ImageClassifier()
    print(emoji_clf.classify(some_emoji))
    emoji_clf.plot_results('test-plot.jpeg')










    
