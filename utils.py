import pandas as pd
import numpy as np
import os
import PIL
import tensorflow as tf

# Joosepi kood:
def onehot(labels: str) -> np.ndarray:
    labels_int = list(map(int, labels.replace('l', '').split(' ')))
    labels = np.zeros(92)
    labels[labels_int] = 1.0
    return labels


def load_train_images_labels():
    df = pd.read_csv('train.csv')
    imgs = []
    img_labels = []

    for img_id, labels in zip(df.image_id.values, df.labels.values):
        # can fail
        try:
            img = PIL.Image.open(os.path.join('images', img_id))
            img.load()

            imgs.append(img)

            label_int = list(map(int, labels.replace('l', '').split(' ')))
            labels = np.zeros(92)
            labels[label_int] = 1.0

            img_labels.append(labels)
        except FileNotFoundError:
            print(img_id, 'doesnt exist')

    img_labels = np.array([i[1] for i in img_labels])

    return imgs, np.array(img_labels)


####### Kea kood ########

def load_test_images():
    df = pd.read_csv('test.csv')
    imgs = []

    for img_id in df.image_id.values:
        
        try:
            
            image = tf.keras.preprocessing.image.load_img(os.path.join('images', img_id),
                    target_size=(300, 300),
                    keep_aspect_ratio = True,
                    color_mode = 'rgb')
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            
            image = np.expand_dims(input_arr, axis=0)
            image = image/255
            image = np.squeeze(image)

            imgs.append(image)

        except FileNotFoundError:
            print(img_id, 'doesnt exist')

    return np.array(imgs)


def load_train_images():
    df = pd.read_csv('train.csv')
    imgs = []
    img_labels = []

    for img_id, labels in zip(df.image_id.values, df.labels.values):
        
        try:
            image = tf.keras.preprocessing.image.load_img(os.path.join('images', img_id),
                    target_size=(300, 300),
                    keep_aspect_ratio = True,
                    color_mode = 'rgb')
            input_arr = tf.keras.preprocessing.image.img_to_array(image)

            image = np.expand_dims(input_arr, axis=0)
            image = image/255
            image = np.squeeze(image)
            imgs.append(image)

            label_int = list(map(int, labels.replace('l', '').split(' ')))
            labels = np.zeros(92)
            labels[label_int] = 1.0

            img_labels.append(labels)
            
        except FileNotFoundError:
            print(img_id, 'doesnt exist')

    return np.array(imgs), img_labels


# Method for loading and one-hot encoding training data with numeric labels
def load_prep_train():
    # Load data
    training_data = pd.read_csv('train.csv')
    labels = pd.read_csv('labels.csv')

    # Create columns for each label and give 0 as a default value for all rows (1 image per 1 row)
    for id, label in enumerate(labels['label_id'].values):
        training_data[label] = 0

    # Go through the 'labels' column values, separate the labels and replace 0 with 1
    # in the newly created labels' columns for each label that is given to that image, others remain 0
    for image_index, labels in enumerate(training_data['labels'].values):
        separated_labels = labels.split(' ')
        for label in separated_labels:
            training_data.at[image_index, label] = 1

    # Drop the old 'labels' column as it is no longer needed
    training_data.drop('labels', axis=1, inplace=True)

    return training_data

# Method to resize images
def resize(images, maxsize = 300):
    train_imgs_resized = []
    for image in images:
        image_resized = image.resize((maxsize,maxsize), PIL.Image.Resampling.LANCZOS)
        background = PIL.Image.new('RGBA', (maxsize, maxsize), (255, 255, 255, 255))
        offset = (round((maxsize - maxsize) / 2), round((maxsize - maxsize) / 2))
        background = background.paste(maxsize, offset)
        image_resized = background.convert('RGB')
        train_imgs_resized.append(np.array(image_resized.convert('RGB')))
    train_imgs_resized = np.array(train_imgs_resized)

    return train_imgs_resized


