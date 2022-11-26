import skimage
import numpy as np
import skimage.morphology as morph
from skimage.segmentation import clear_border
from skimage import filters
import keras
from keras.layers import Dense
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator
import glob
from typing import Literal
import re
from keras.applications import EfficientNetB4
import tensorflow_addons as tfa
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

IMAGE_SIZE = 128
BATCH_SIZE = 64


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
    total = binary_mask.sum()
    coef = 0.06
    total_light = 0
    while total_light < (total * 0.5):
        coef -= 0.01
        binary_mask_light = morph.remove_small_objects(
            binary_mask, coef * binary_mask.shape[0] * binary_mask.shape[1]
        )
        total_light = binary_mask_light.sum()
    binary_mask = binary_mask_light
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
    efficient_net = EfficientNetB4(
        weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    x = efficient_net.output
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(4, activation="sigmoid")(x)
    model = Model(inputs=efficient_net.input, outputs=predictions)
    model.compile(
        optimizer="adam",
        loss=tfa.losses.SigmoidFocalCrossEntropy(),
        metrics=["accuracy"],
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def preprocess_extract_patch():
    def _preprocess_extract_patch(x):
        img = prepare_image(x.astype(np.uint8))
        return img

    return _preprocess_extract_patch


def prepare_labels(dataset: Literal["Train", "Test"]):
    data_f = glob.glob("dataset/" + dataset + "/*/*")
    df = pd.DataFrame(
        [re.split(r"[/\\]", i)[2:] for i in data_f], columns=["class", "imFile"]
    )
    df["repo"] = [i[8:] for i in data_f]
    dict_class = {
        "castanea": ["dente", "alterne", "simple", "oui"],
        "convolvulaceae": ["lisse", "alterne", "simple", "non"],
        "magnolia": ["lisse", "alterne", "simple", "oui"],
        "ulmus": ["dente", "alterne", "simple", "oui"],
        "litsea": ["lisse", "alterne", "simple", "oui"],
        "laurus": ["lisse", "oppose", "simple", "oui"],
        "monimiaceae": ["lisse", "oppose", "simple", "oui"],
        "desmodium": ["lisse", "alterne", "composee", "non"],
        "amborella": ["lisse", "alterne", "simple", "oui"],
        "eugenia": ["lisse", "oppose", "simple", "oui"],
        "rubus": ["dente", "alterne", "composee", "oui"],
    }
    for i in dict_class.keys():
        df.loc[
            df["class"] == i, ["bord", "phyllotaxie", "typeFeuille", "ligneux"]
        ] = dict_class[i]
    df.to_csv(dataset + "_labels.csv", index=False)


def train_model(model):
    df = pd.read_csv("Train_labels.csv")

    df = pd.get_dummies(
        df, columns=["bord", "phyllotaxie", "typeFeuille", "ligneux"], drop_first=True
    )
    preprocessing_function = preprocess_extract_patch()

    datagen_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
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

    test_df = pd.read_csv("Test_labels.csv")

    test_df = pd.get_dummies(
        test_df,
        columns=["bord", "phyllotaxie", "typeFeuille", "ligneux"],
        drop_first=True,
    )

    test_generator = datagen_aug.flow_from_dataframe(
        dataframe=test_df,
        directory="dataset",
        x_col="repo",
        y_col=columns,
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=1),
        keras.callbacks.ModelCheckpoint(
            filepath="model_checkpoint",
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        ModelCheckpoint(
            "model.hdf5", save_best_only=True, verbose=0, monitor="val_loss", mode="min"
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=0.000001, verbose=1
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        verbose=1,
        steps_per_epoch=train_generator.n / BATCH_SIZE,
        callbacks=[callbacks],
    )

    preds = model.predict(test_generator, steps=test_generator.n / BATCH_SIZE)

    return history, preds


def print_confusion_matrix(
    confusion_matrix, axes, class_label, class_names, fontsize=14
):

    df_cm = pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    axes.set_ylabel("True label")
    axes.set_xlabel("Predicted label")
    axes.set_title("Confusion Matrix for the class - " + class_label)


def print_multilabel_confusion_matrix(y_preds):
    df = pd.read_csv("Test_labels.csv")

    df = pd.get_dummies(
        df, columns=["bord", "phyllotaxie", "typeFeuille", "ligneux"], drop_first=True
    )
    y_preds = y_preds.round().astype(np.uint8)
    y_test = np.array(df.iloc[:, 3:])
    y_test.shape, y_preds.shape
    confusion_matrix = multilabel_confusion_matrix(y_test, y_preds)
    labels = ["bord_lisse", "phyllotaxie_oppose", "typeFeuille_simple", "ligneux_oui"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, labels):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.show()
