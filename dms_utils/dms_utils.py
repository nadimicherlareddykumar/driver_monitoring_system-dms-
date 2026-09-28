from glob import glob
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
import collections


ACTIONS = {'phonecall': 0, 'texting': 1}
AUTOTUNE = 4

def preprocess_image(image, size=(224, 224)):
    """ Preprocess image for training/inference. """
    if isinstance(size, list):
        size = tuple(size)
    return cv2.resize(image, size)

def load_and_preprocess_image(path, size=(224, 224)):
    """ Load and preprocess image for training/inference. """
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return preprocess_image(image, size)
    return None


# The DMD dataset, collected from 5 persons, each with 4 different scenes
SESSIONS = {
    1: [
        '2019-03-08-09:31:15',
        '2019-03-08-09:21:03',
        '2019-03-14-14:31:08',
        '2019-03-22-11:49:58'],
    2: [
        '2019-03-08-10:01:44',
        '2019-03-08-09:50:49',
        '2019-03-13-09:23:42',
        '2019-03-22-09:15:55'],
    3: [
        '2019-03-08-10:27:38',
        '2019-03-08-10:16:48',
        '2019-03-13-09:41:01',
        '2019-03-22-10:15:27'],
    4: [
        '2019-03-13-10:36:15',
        '2019-03-13-10:43:06',
        '2019-03-13-11:00:49',
        '2019-03-25-11:44:29'],
    5: [
        '2019-03-08-10:57:00',
        '2019-03-08-10:46:46',
        '2019-03-13-09:10:35',
        '2019-03-22-11:28:00']
}

def sampling_data(rootdir, num_skip_frames=10):
    """ Read dmd data paths and labels from a root directory. """

    all_image_paths  = None
    all_image_labels = None

    for subdir, dirs, files in os.walk(rootdir):
        for action in list(ACTIONS.keys()):
            if action in subdir:
                image_dir = subdir + '/*.jpg'
                image_paths = np.sort(np.array(glob(image_dir)))
                image_paths = image_paths[10::num_skip_frames]
                image_labels = np.array([ACTIONS[action]] * image_paths.shape[0])

                all_image_paths = np.concatenate((all_image_paths,image_paths), axis=0) \
                    if all_image_paths is not None else image_paths
                all_image_labels = np.concatenate((all_image_labels,image_labels), axis=0) \
                    if all_image_labels is not None else image_labels

    if all_image_paths is None:
        return np.array([], dtype=str), np.array([], dtype=np.int64)

    assert all_image_labels.shape[0] == all_image_paths.shape[0]

    return all_image_paths, all_image_labels

def plot_sample_distribution(labels):
    """ Plot sample distribution per class. """

    classes, cnts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(12, 1))
    plt.barh(list(ACTIONS.keys()), cnts, height=0.6)
    for i, v in enumerate(cnts):
        plt.text(v, i, ' '+str(v), va='center')
    plt.xlabel('Counts')
    plt.title("Distribution of samples")

def preprocess_image_tf(image, size):
    """ Preprocess image for training using TensorFlow. """

    import tensorflow as tf
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)

    return image

def load_and_preprocess_image_tf(path, size=[224,224]):
    """ Load and preprocess image for training using TensorFlow. """

    import tensorflow as tf
    image = tf.io.read_file(path)
    return preprocess_image_tf(image, size)

def display4images(images, labels):
    """ Display 4 image samples. """

    plt.figure(figsize=(8,8))
    for n, image in enumerate(images):
        plt.subplot(2,2,n+1)
        plt.imshow(load_and_preprocess_image_tf(image)/255.0)
        plt.grid(False)
        plt.title(list(ACTIONS.keys())[labels[n]].title())
        plt.xticks([])
        plt.yticks([])
    plt.show()

def create_ds(paths, labels):
    """ Create a tf dataset for training. """

    import tensorflow as tf
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(load_and_preprocess_image_tf, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds

def plot_his_metrics(history, image_name):
    """ Plot the loss and accuracy of training. """

    metrics = ['loss', 'accuracy']
    plt.figure(figsize=(15, 6))
    plt.rc('font', size=12)
    for n, metric in enumerate(metrics):
        name = metric.capitalize()
        plt.subplot(1,2,n+1)
        plt.plot(history.epoch, history.history[metric], label='Training', lw=3, color='navy')
        plt.plot(history.epoch, history.history['val_'+metric], lw=3, label='Validation', color='deeppink')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title('Model '+name)
        plt.legend()
    plt.savefig(image_name)
