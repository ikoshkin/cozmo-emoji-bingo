from keras import models
import numpy as np
import h5py
from keras.preprocessing.image import img_to_array, load_img
import os
import random
import cv2


def preprocess_img(img_array):
        return (img_array / 255.)


def load_model_and_weights():
    model = models.load_model('./output-aws/vgg16_v0.h5')
    model.load_weights('./output-aws/hdf5/vgg16_v0-015-0.006.hdf5')
    return model

def pick_random_image():
    labels = ['human', 'robot']
    label = random.choice(labels)
    
    path = os.path.join('./images/originals', label)
    fname = random.choice(os.listdir(path))
    
    return label, os.path.join(path, fname), fname


if __name__ == '__main__':

    model = load_model_and_weights()

    label, img_file, img_id = pick_random_image()

    print("[INFO] loading and pre-processing image...")
    image = load_img(img_file, target_size=(320, 240))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_img(image)

    # classify the image
    print("[INFO] classifying image ...")
    pred_prob = model.predict(image)
    pred_label = model.predict_classes(image)
    orig = cv2.imread(img_file)

    cv2.putText(orig, "Label: {}, {}%".format(pred_label, pred_prob * 100),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite("./testpred/pred-{}".format(img_id), orig)








    
