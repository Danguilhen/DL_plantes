# from ipoly import load
import skimage
import numpy as np
import skimage.morphology as morph
from skimage.segmentation import clear_border
from skimage import filters
from PIL.Image import fromarray as afficher
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 500


def croper(image: np.array, margin: int = 3):
    if len(np.unique(image)) == 1:
        raise Exception("The image is composed of a single color.")
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


def prepare_image(image: np.array):
    gim = skimage.color.rgb2gray(image)
    threshold = filters.threshold_otsu(gim)
    binary_mask = gim < threshold
    binary_mask = morph.remove_small_objects(
        binary_mask, 0.03 * binary_mask.shape[0] * binary_mask.shape[1]
    )  # réduire le coefficient si le nombre de pixel diminue
    binary_mask_cleared = clear_border(
        skimage.morphology.remove_small_holes(binary_mask, 300)
    )
    if binary_mask_cleared.sum() > binary_mask.sum() * 0.3:
        binary_mask = binary_mask_cleared
    labeled_image, _ = skimage.measure.label(binary_mask, return_num=True)
    image[labeled_image == 0] = 255
    img = croper(image)
    scale_percent = min(
        IMAGE_SIZE * 100 / img.shape[0], IMAGE_SIZE * 100 / img.shape[1]
    )
    width = int(round(img.shape[1] * scale_percent / 100))
    height = int(round(img.shape[0] * scale_percent / 100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if resized.shape[1] != IMAGE_SIZE:
        white = np.full(
            (IMAGE_SIZE, IMAGE_SIZE - resized.shape[1], 3), 255, dtype=np.uint8
        )
        result = np.concatenate((resized, white), axis=1)
    else:
        white = np.full(
            (IMAGE_SIZE - resized.shape[0], IMAGE_SIZE, 3), 255, dtype=np.uint8
        )
        result = np.concatenate((resized, white), axis=0)
    return result


def create_model():
    model = keras.models.Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
    )
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(4, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def preprocess_extract_patch():
    def _preprocess_extract_patch(x):
        img = prepare_image(x.astype(np.uint8))
        return img

    return _preprocess_extract_patch


def train_model(model):
    df = pd.read_csv("labels.csv")

    df = pd.get_dummies(
        df, columns=["bord", "phyllotaxie", "typeFeuille", "ligneux"], drop_first=True
    )
    preprocessing_function = preprocess_extract_patch()

    BATCH_SIZE = 5

    datagen_aug = ImageDataGenerator(
        preprocessing_function=preprocessing_function, rotation_range=25
    )

    columns = ["bord_lisse", "phyllotaxie_oppose", "typeFeuille_simple", "ligneux_oui"]
    train_generator = datagen_aug.flow_from_dataframe(
        dataframe=df[: round(df.shape[0] * 0.8)],
        directory="dataset",
        x_col="repo",
        y_col=columns,
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    val_generator = datagen_aug.flow_from_dataframe(
        dataframe=df[round(df.shape[0] * 0.8) :],
        directory="dataset",
        x_col="repo",
        y_col=columns,
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    callback = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
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
        validation_data=val_generator,
        # steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=10,
        verbose=1,
        callbacks=[callback],
    )
