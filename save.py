import os
import glob
import shutil

import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import tensorflow as tf

# Import deep learning package (tensorflow)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Set seed nunmber to all packages
seed_number = 24
np.random.seed(seed_number)
tf.random.set_seed(seed_number)


if __name__ == "__main__":
    # Configuring directories
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
    root = "../input"
    input_dir = os.path.join(root, "LCC_FASD")
    train_dir = os.path.join(input_dir, 'LCC_FASD_training')
    val_dir = os.path.join(input_dir, 'LCC_FASD_development')
    test_dir = os.path.join(input_dir, 'LCC_FASD_evaluation')

    dataset_dir = [dir for dir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, dir))]
    label_name = [subdir for subdir in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, subdir))]

    # Printing the directory informations
    print(f"Main directories\t: {os.listdir(root)}")
    print(f"Dataset sub-directories\t: {dataset_dir}")
    print(f"Train set directory\t: {label_name}")

    # ----------------------- HANDLING DATASET -----------------------
    # Define image size
    img_width, img_height = 224, 224

    # Instantiate data generator for training procedure
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_gen = test_datagen.flow_from_directory(test_dir,
                                                batch_size=1,
                                                class_mode='binary',
                                                target_size=(img_width, img_height),
                                                seed=seed_number,
                                                shuffle=False)

    # Displaying the dataset generator information
    print(f'Test set batch shape\t: {next(test_gen)[0].shape}')

    # ----------------------- GENERATE MODEL -----------------------
    # Don't forget to turn on the Internet to download the respective pre-trained weights!
    pretrain_net = mobilenet_v2.MobileNetV2(input_shape=(img_width, img_height, 3),
                                            include_top=False,
                                            weights='imagenet')

    # ------ Freezing layer(s) up to a specific layer ------
    freeze_before = "block_16_expand"  # use None to train, use "all" to freeze all the layers

    if freeze_before:
        for layer in pretrain_net.layers:
            if layer.name == freeze_before:
                break
            else:
                layer.trainable = False

    # Adding extra layer for our problem
    x = pretrain_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=128, activation='relu', name='extra_fc1')(x)
    x = Dropout(rate=0.2, name='extra_dropout1')(x)
    x = Dense(1, activation='sigmoid', name='classifier')(x)

    model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv2_spoof')
    # print(model.summary())

    # ----------------------- LOADING MODEL -----------------------
    # Input filename
    train_id = "lcc-train02"
    best_filepath = "mobilenetv2-best.hdf5"
    infer_filepath = f"{train_id}-infer.hdf5"

    # Load the saved weights
    savedir = os.path.join("../output", train_id)
    weights_path = os.path.join(savedir, best_filepath)
    assert os.path.isfile(weights_path)

    print(f"Load weights from: '{weights_path}'")
    model.load_weights(weights_path)

    # ----------------------- RESULTS EVALUATION -----------------------
    test_length = len(test_gen.classes)

    # Calculate prediction
    threshold = 0.5  # Define the sigmoid threshold for True or False
    y_pred_value = np.squeeze(model.predict(test_gen, steps=test_length, verbose=1))
    y_pred = np.zeros(y_pred_value.shape)

    y_pred[y_pred_value > threshold] = 1
    y_true = test_gen.classes

    # Sanity check on the y_pred and y_true value
    print(f"Label\t\t: {y_true[:10]}")
    print(f"Prediction\t: {y_pred[:10]}")

    # Confusion matrix result
    confusion_matrix_result = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(confusion_matrix_result,
                          figsize=(12, 8),
                          hide_ticks=True,
                          cmap=plt.cm.jet)
    plt.title("Face Spoofing Detection")
    plt.xticks(range(2), ['Real', 'Spoof'], fontsize=16)
    plt.yticks(range(2), ['Real', 'Spoof'], fontsize=16)
    plt.show()

    # Precision and Recall metrics
    tn, fp, fn, tp = confusion_matrix_result.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print("Report Summary:")
    print("Precision\t: {:.2f}%".format(precision * 100))
    print("Recall\t\t: {:.2f}%".format(recall * 100))
    print("F1 Score\t: {:.2f}%".format(f1_score * 100))

    print("\nNotes: ")
    print("True labels\t: Spoof")
    print("False labels\t: Real")

    # ----------------------- SAVING THE MODEL -----------------------
    # Save the model
    modeldir = os.path.join("../output", "infer")
    if not os.path.isdir(modeldir):
        os.makedirs(modeldir)

    model_path = os.path.join(modeldir, infer_filepath)
    print(f"Save full model at: '{model_path}'")
    model.save(model_path)
