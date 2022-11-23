from ipoly import load
import skimage
import numpy as np
import skimage.morphology as morph
from skimage.segmentation import clear_border
from skimage import filters
from PIL.Image import fromarray as  afficher
import keras
from keras.layers import Dense, Conv2D
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def croper(image, margin=3):
    if len(image.shape) == 3:
        image_sum = image.sum(axis=2)%765
    else: image_sum = image == 0
    true_points = np.argwhere(image_sum)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    return image[max(0, top_left[0]-margin):bottom_right[0]+1+margin,
           max(0, top_left[1]-margin):bottom_right[1]+1+margin]
    
def prepare_image(image):
        threshold = filters.threshold_otsu(skimage.color.rgb2gray(image))
        gim = skimage.color.rgb2gray(image)
        binary_mask = gim < threshold
        binary_mask = morph.remove_small_objects(binary_mask, 50000)
        binary_mask_cleared = clear_border(skimage.morphology.remove_small_holes(binary_mask, 300))
        if binary_mask_cleared.sum() > binary_mask.sum()*0.3:
                binary_mask = binary_mask_cleared
        labeled_image, count = skimage.measure.label(binary_mask, return_num=True)
        image[labeled_image == 0] = 255
        img = croper(image)
        return img
    
def prepare_dataset():
    df = load("labels.csv")
    df = pd.get_dummies(df, columns=['bord', 'phyllotaxie', 'typeFeuille', 'ligneux'], drop_first=True)
    X_train = list()
    Y_train = list()
    for row in df.iterrows():
        print('dataset/' + row[1].repo)
        i = load('dataset/' + row[1].repo)
        img = prepare_image(i)
        scale_percent = 500 * 100 / img.shape[0]
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        Y_train.append(list(row[1][3:7]))
        X_train.append(resized)
    return X_train, Y_train
    
def create_model():
    model = keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=5, strides=2, activation='relu', input_shape=(268, 182, 3)))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))   # Final Layer using Softmax

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(X_train, Y_train, model):
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    batch_size = 64

    x = np.array(X_train)
    y = np.array(Y_train)

    x = aug.flow(x, y, batch_size=batch_size)
    
    H = model.fit(
        x=x,
        #validation_data=(X_test, Y_test),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=5, verbose=1)
