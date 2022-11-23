#from ipoly import load
import skimage
import numpy as np
import skimage.morphology as morph
from skimage.segmentation import clear_border
from skimage import filters
from PIL.Image import fromarray as afficher
import keras
from keras.layers import Dense, Conv2D,MaxPooling2D
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator


def croper(image, margin=3):
    if len(image.shape) == 3:
        image_sum = image.sum(axis=2) % 765
    else:
        image_sum = image == 0
    true_points = np.argwhere(image_sum)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    return image[
        max(0, top_left[0] - margin) : bottom_right[0] + 1 + margin,
        max(0, top_left[1] - margin) : bottom_right[1] + 1 + margin,
    ]

def prepare_image(image):
    gim = skimage.color.rgb2gray(image)
    threshold = filters.threshold_otsu(gim)
    binary_mask = gim < threshold
    binary_mask = morph.remove_small_objects(binary_mask, 50000)
    binary_mask_cleared = clear_border(
        skimage.morphology.remove_small_holes(binary_mask, 300)
    )
    if binary_mask_cleared.sum() > binary_mask.sum() * 0.3:
        binary_mask = binary_mask_cleared
    labeled_image, _ = skimage.measure.label(binary_mask, return_num=True)
    image[labeled_image == 0] = 255
    img = croper(image)
    scale_percent = min(500 * 100 / img.shape[0], 500 * 100 / img.shape[1])
    width = int(round(img.shape[1] * scale_percent / 100))
    height = int(round(img.shape[0] * scale_percent / 100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if resized.shape[1] != 500:
        white = np.full((500, 500 - resized.shape[1], 3), 255, dtype=np.uint8)
        result = np.concatenate((resized, white), axis=1)
    else:
        white = np.full((500 - resized.shape[0], 500, 3), 255, dtype=np.uint8)
        result = np.concatenate((resized, white), axis=0)
    return result


def prepare_dataset():
    df = load("labels.csv")
    df = pd.get_dummies(
        df, columns=["bord", "phyllotaxie", "typeFeuille", "ligneux"], drop_first=True
    )
    X_train = list()
    Y_train = list()
    for row in df.iterrows():
        print("dataset/" + row[1].repo)
        i = load("dataset/" + row[1].repo)
        result = prepare_image(i)
        Y_train.append(list(row[1][3:7]))
        X_train.append(result)
    return X_train, Y_train


def create_model():
    model = keras.models.Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(500, 500, 3),
        )
    )
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(4, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def preprocess_extract_patch():
    def _preprocess_extract_patch(x):
        img = prepare_image(x)
        return img
    return _preprocess_extract_patch



def train_model( model):
    df=pd.read_csv("labels.csv")

    df = pd.get_dummies(
        df, columns=["bord", "phyllotaxie", "typeFeuille", "ligneux"], drop_first=True
    )
    preprocessing_function = preprocess_extract_patch()

    BATCH_SIZE=5

    train_datagen_aug = ImageDataGenerator(preprocessing_function=preprocessing_function,rotation_range=25)
    columns=["bord_lisse"	,"phyllotaxie_oppose"	,"typeFeuille_simple"	,"ligneux_oui"]
    train_generator=train_datagen_aug.flow_from_dataframe(
    dataframe=df[:5],
    directory="dataset",
    x_col="repo",
    y_col=columns,
    batch_size=5,
    seed=42,
    shuffle=True,
    class_mode="raw")




    callback = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=3),
        keras.callbacks.ModelCheckpoint(
            filepath="model_checkpoint",
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    H = model.fit(
        train_generator,
        #validation_data=(X_test, Y_test),
        #steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=1,
        verbose=1,
        callbacks=[callback],
    )
