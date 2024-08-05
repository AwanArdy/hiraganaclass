import numpy as np
import os

def load_and_preprocess_data(train_imgs_filename, train_labels_filename, test_imgs_filename, test_labels_filename):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    x_train = np.load(os.path.join(root_dir, train_imgs_filename))['arr_0']
    y_train = np.load(os.path.join(root_dir, train_labels_filename))['arr_0']

    y_test = np.load(os.path.join(root_dir, test_imgs_filename))['arr_0']
    y_test = np.load(os.path.join(root_dir, test_labels_filename))['arr_0']

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x