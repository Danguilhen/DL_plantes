import skimage
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import skimage.morphology as morph
from skimage.segmentation import clear_border
from skimage import filters
import keras
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator
import glob
from typing import Literal
import re
from keras.applications import EfficientNetB4
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import class_weight
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications.densenet import DenseNet121
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report


import warnings

warnings.filterwarnings("ignore")


class ai_plantes:
    def __init__(self, IMAGE_SIZE: int = 320, BATCH_SIZE: int = 10):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE

    def croper(self, image: np.array, margin: int = 18):
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

    def prepare_image(self, image: np.array):
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
        img = self.croper(image)
        scale_percent = min(
            self.IMAGE_SIZE * 100 / img.shape[0], self.IMAGE_SIZE * 100 / img.shape[1]
        )
        width = int(round(img.shape[1] * scale_percent / 100))
        height = int(round(img.shape[0] * scale_percent / 100))
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if resized.shape[1] != self.IMAGE_SIZE:
            white = np.full(
                (self.IMAGE_SIZE, self.IMAGE_SIZE - resized.shape[1], 3),
                255,
                dtype=np.uint8,
            )
            result = np.concatenate((resized, white), axis=1)
        else:
            white = np.full(
                (self.IMAGE_SIZE - resized.shape[0], self.IMAGE_SIZE, 3),
                255,
                dtype=np.uint8,
            )
            result = np.concatenate((resized, white), axis=0)
        return result

    def create_model(
        self,
        allow_train_OnAll=False,
        architecture=DenseNet121,
        sorce="DenseNet121.h5",
        balance=True,
        weights=None,
    ):
        """
        This function will cearte a model with a model with a given architecture.

        Parameters
        ----------
        dont_allow_train_OnAll : bool
            if True use the pre trained model
        architecture : classe
            the architecture of the chossen model

        sorce : string
            The path of your pretrained weights
        balance : bool
            to belance the clases for sigmoid output
        weights : string
            With or without wights from the orginal model (or you can choose "imagenet" )

        Returns
        -------
        No thing
        """
        The_model = architecture(  # EfficientNetB4(
            weights=None,
            include_top=False,
            input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
        )
        x = The_model.output
        x = GlobalAveragePooling2D()(x)  # adding global layer
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        #####################################################

        # import pretrained weights
        if sorce == "DenseNet121.h5":  # pretrained model on plants
            preds = Dense(4, activation="softmax")(x)
            model = Model(inputs=The_model.input, outputs=preds)
            model.load_weights(sorce)  # here are the weights
            predictions = Dense(len(self.columns), activation=self.out_put)(
                model.layers[-2].output
            )
            self.model = Model(inputs=model.input, outputs=predictions)
        else:
            predictions = Dense(len(self.columns), activation=self.out_put)(x)
            self.model = Model(inputs=The_model.input, outputs=predictions)
        #####################################################

        # pre-trained  choice
        if not allow_train_OnAll:
            for layer in self.model.layers[:-8]:
                layer.trainable = False
            for layer in self.model.layers[-8:]:
                layer.trainable = True
        #####################################################

        # loss function choice
        loss = "binary_crossentropy"  # tfa.losses.SigmoidFocalCrossEntropy(),
        if (self.out_put == "sigmoid") & (balance):
            loss = self.get_weighted_loss(self.pos_weights, self.neg_weights)
        elif self.out_put == "softmax":
            loss = "CategoricalCrossentropy"
        #####################################################

        self.model.compile(
            optimizer="adam",
            loss=loss,
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

    def preprocess_extract_patch(self):
        def _preprocess_extract_patch(x):
            img = self.prepare_image(x)
            return img

        return _preprocess_extract_patch

    # ----------------------------------------------------------------------------------------------------------------------------------
    def compute_class_freqs(self, labels):
        """
        This function will help to calculate the wights of negative and postive of each class
        Input :
            - labels : are columns names
        Output :
            - positive_frequencies
            - negative_frequencies
        """
        N = labels.shape[0]
        positive_frequencies = np.sum(labels == 1, axis=0) / N
        negative_frequencies = np.sum(labels == 0, axis=0) / N
        return positive_frequencies, negative_frequencies

    # ----------------------------------------------------------------------------------------------------------------------------------

    def get_weighted_loss(self, pos_weights, neg_weights, epsilon=1e-7):
        """
        This function will calculate the loss function of the model
        Input :
            - pos_weights : positive frequencies wights
            - neg_weights : negative frequencies wights
            - epsilon : to not devide by 0
        Output :
            - loss : the loss classic function for the lost function.
        """

        def weighted_loss(y_true, y_pred):
            y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
            # initialize loss to zero
            loss = 0.0

            for i in range(len([pos_weights])):
                # for each class, add average weighted loss for that class
                loss += K.mean(
                    -(
                        (pos_weights * y_true * K.log(y_pred + epsilon))
                        + (neg_weights * (1 - y_true) * K.log(1 - y_pred + epsilon))
                    )
                )  # complete this line
            return loss

        return weighted_loss

    # ----------------------------------------------------------------------------------------------------------------------------------

    def prepare_labels(self, dataset: Literal["Train", "Test"]):
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

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def preporcess(
        self,
        data_file="Train_labels.csv",
        isMacOs=False,
        out_put="sigmoid",
        min_sample=False,
    ):
        """
        This function will import dataFrames, split the data, tranforme the data and
         preporcess the data. It will creat 3 data base Trai - Valdidation - Test
         all of them will have the forme of gernators.

        Parameters
        ----------
        data_file : string
            train data in csv file
        isMacOs : bool
            True if Mac OS user read path images

        out_put : string
            What kind of output sigmoid or softmax
        min_sample : bool
            It will blance the data in equal distubatioin
        Returns
        -------
        no thing

        """
        # Data Frame prepreprations
        # Importaning Data
        print("We are going to preporcess the data ...")
        self.df = pd.read_csv(data_file)
        self.test_df = pd.read_csv("Test_labels.csv")
        self.out_put = out_put
        # creating a labels variable to groupe all the output together
        self.df["labels"] = (
            self.df.bord
            + "-"
            + self.df.phyllotaxie
            + "-"
            + self.df.typeFeuille
            + "-"
            + self.df.ligneux
        )
        self.test_df["labels"] = (
            self.test_df.bord
            + "-"
            + self.test_df.phyllotaxie
            + "-"
            + self.test_df.typeFeuille
            + "-"
            + self.test_df.ligneux
        )
        self.columns = ["bord", "phyllotaxie", "typeFeuille", "ligneux"]
        if min_sample:
            print("\n\nThe distubation(Disproportionate Sampling methode) is: ")
            uni = np.unique(self.df.labels, return_counts=True)
            m = [
                print(f"Classe {cla} is repeated {n} times ")
                for cla, n in zip(uni[0], uni[1])
            ]
            print(
                "We have to make them all equal to the minimum count to balance the distrubation which is:  ",
                min(uni[1]),
            )
            self.df = self.df.groupby("labels", group_keys=False).apply(
                lambda x: x.sample(min(uni[1]))
            )
            print("##############################################################\n\n")
        ##################################################################

        if self.out_put == "softmax":
            # if softmax add compute weights
            self.columns = ["labels"]
        # transofrm oneHot_encoding
        self.pre_classe = OneHotEncoder(drop="if_binary")
        seg = self.pre_classe.fit_transform(self.df[self.columns]).toarray()
        seg_test = self.pre_classe.transform(self.test_df[self.columns]).toarray()
        # new columns
        self.columns = self.pre_classe.get_feature_names_out()
        self.df[self.columns] = seg
        self.test_df[self.columns] = seg_test
        ##################################################################

        if isMacOs:
            # for Mac OS useres to read files
            self.df.repo = self.df.repo.str.replace("\\", "/", regex=False)
            self.test_df.repo = self.test_df.repo.str.replace("\\", "/", regex=False)
        ##################################################################

        # split train validation data
        X_train, X_test, y_train, y_test = train_test_split(
            self.df["repo"],
            self.df[self.columns],
            test_size=0.10,
            random_state=42,
            stratify=self.df["labels"],
        )
        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_test, y_test], axis=1)
        ##################################################################

        # start augmatation here
        self.datagen_aug = ImageDataGenerator(
            preprocessing_function=self.preprocess_extract_patch(),
            rescale=1.0 / 255,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # zoom_range=0.2,
            # brightness_range=[0.4,1.5],
            # horizontal_flip=True,
            # vertical_flip=True,
        )
        ##################################################################

        self.train_generator = self.datagen_aug.flow_from_dataframe(
            dataframe=df_train,  # df[: round(df.shape[0] * 0.8)],
            directory="dataset",
            x_col="repo",
            y_col=self.columns,
            batch_size=self.BATCH_SIZE,
            seed=42,
            shuffle=True,
            class_mode="raw",
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
        )
        print("The train_generator is ready ")
        ##################################################################

        self.val_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            preprocessing_function=self.preprocess_extract_patch(),
        )
        self.val_generator = self.val_datagen.flow_from_dataframe(
            dataframe=df_val,  # df[round(df.shape[0] * 0.8) :],
            directory="dataset",
            x_col="repo",
            y_col=self.columns,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            class_mode="raw",
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
        )
        print("The val_generator is ready. ")
        ##################################################################

        # compute loss:
        if self.out_put == "sigmoid":
            freq_pos, freq_neg = self.compute_class_freqs(self.train_generator.labels)
            self.pos_weights, self.neg_weights = tf.cast(freq_neg, tf.float32), tf.cast(
                freq_pos, tf.float32
            )
        ##################################################################

        self.test_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            preprocessing_function=self.preprocess_extract_patch(),
        )

        self.test_generator = self.test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            directory="dataset",
            x_col="repo",
            y_col=self.columns,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            class_mode="raw",
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
        )
        print("The test_generator is ready. ")

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def fit(self, epochs=10):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath="model_checkpoint",
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            ),
            ModelCheckpoint(
                "model.hdf5",
                save_best_only=True,
                verbose=0,
                monitor="val_loss",
                mode="min",
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=5, min_lr=0.000001, verbose=1
            ),
        ]

        history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            verbose=1,
            steps_per_epoch=self.train_generator.n / self.BATCH_SIZE,
            callbacks=[callbacks],
            # class_weight=self.class_weights if (self.out_put == "softmax" ) else None
        )

        self.preds = self.model.predict(
            self.test_generator, steps=len(self.test_generator)
        )

        return history, self.preds

    def print_confusion_matrix(
        self, confusion_matrix, axes, class_label, class_names, fontsize=14
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

    def print_multilabel_confusion_matrix(self):

        y_preds = np.rint(self.preds)  # .round().astype(np.uint8)
        y_test = np.array(self.test_df[self.columns])
        print(classification_report(y_preds, y_test, zero_division=1))
        confusion_matrix = multilabel_confusion_matrix(y_test, y_preds)
        labels = [
            "bord_lisse",
            "phyllotaxie_oppose",
            "typeFeuille_simple",
            "ligneux_oui",
        ]
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, labels):
            self.print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

        fig.tight_layout()
        plt.show()

    def get_roc_curve(self):
        # this function will plot the ROC curve for each class
        labels = self.columns
        auc_roc_vals = []
        for i in range(len(labels)):
            try:
                gt = self.test_generator.labels[:, i]
                pred = self.preds[:, i]
                auc_roc = roc_auc_score(gt, pred)
                auc_roc_vals.append(auc_roc)
                fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
                plt.figure(1, figsize=(12, 10))
                plt.plot([0, 1], [0, 1], "k--")
                plt.plot(
                    fpr_rf,
                    tpr_rf,
                    label=labels[i] + " (" + str(round(auc_roc, 3)) + ")",
                )
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.title("ROC curve")
                plt.legend(loc="best")
            except:
                print(
                    f"Error in generating ROC curve for {labels[i]}. "
                    f"Dataset lacks enough examples."
                )
        plt.show()
        return auc_roc_vals
