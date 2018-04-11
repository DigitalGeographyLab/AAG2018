# -*- coding: utf-8 -*-

# Import packages
from keras.applications.nasnet import preprocess_input
from keras.optimizers import Adam
from keras.utils.multi_gpu_utils import multi_gpu_model
from networks import CaptionClassifier, NASNetL, NASNetL_mm, NASNetL_mm_F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import argparse
import os
import numpy as np
import tensorflow as tf


# Set up argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-dd", "--data_dir", required=True,
                help="Path to the file where the data and labels are stored.")
ap.add_argument("-m", "--mode", required=True, type=str,
                help="Type of classifier to train.")
ap.add_argument("-k", "--kfolds", required=True, type=int,
                help="The number of random training / testing splits.")

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
path_to_data = args['data_dir']
folds = args['kfolds']
mode = args['mode']

# Generate random integers between 0 and 10000
seeds = np.random.randint(0, 10000, folds)

# Check what kinds of inputs are required for classification
if mode == 'text':

    # Load word embeddings and labels
    texts = np.load(os.path.join(path_to_data, 'embeddings.npy'))
    labels = np.load(os.path.join(path_to_data, 'labels.npy'))

if mode == 'image':

    # Load images and labels
    images, labels = np.load(os.path.join(path_to_data, 'data.npy')), \
                     np.load(os.path.join(path_to_data, 'labels.npy'))

    # Preprocess images
    images = preprocess_input(images)

if mode in ['early', 'joint']:

    # Load images, word embeddings and labels
    images = np.load(os.path.join(path_to_data, 'data.npy'))
    texts = np.load(os.path.join(path_to_data, 'embeddings.npy'))
    labels = np.load(os.path.join(path_to_data, 'labels.npy'))

    # Preprocess images
    images = preprocess_input(images)

# Calculate balanced class weights and cast into dictionary
class_weight = compute_class_weight('balanced', np.unique(labels), labels)
class_weight = dict(enumerate(class_weight))

# Set up list to hold the classification reports
reports = {}

# Begin looping over the folds
for i, s in enumerate(seeds, start=1):

    # Print status
    print("Now working on fold {}/{}.".format(i, len(seeds)))

    # Build a text classifier if requested
    if mode == 'text':

        # Split the texts and labels into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(texts, labels,
                                                            test_size=0.1,
                                                            random_state=s)

        # Use CPU to compile the model
        with tf.device('/cpu:0'):

            # Build the model
            model = CaptionClassifier.build(neurons=1, l2_penalty=0.000001,
                                            dropout=0.5, activation='sigmoid')

        # Parallelize the model
        model = multi_gpu_model(model, gpus=None)

        # Set up optimizer
        optimizer = Adam()

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['binary_accuracy'])

        # Train the parallel model using GPUs
        training_history = model.fit(X_train, y_train,
                                     epochs=40,
                                     batch_size=128,
                                     shuffle=True,
                                     validation_split=0.1,
                                     class_weight=class_weight,
                                     verbose=1)

        with tf.device('/cpu:0'):

            # Make predictions and convert labels from one hot to sparse
            y_pred = model.predict(X_test, batch_size=64)

            # Round output from sigmoid activation
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            # Get the classification report
            report = classification_report(y_test,
                                           y_pred,
                                           target_names=['activity',
                                                         'not_activity']
                                           )

            # Add classification report to the dictionary of reports
            reports[i] = report

    # Build an image classifier if requested
    if mode == 'image':

        # Split the images and labels into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels,
                                                            test_size=0.1,
                                                            random_state=s)

        # Use CPU to compile the model
        with tf.device('/cpu:0'):

            # Build the model
            model = NASNetL.build(neurons=1, l2_penalty=0.000001, dropout=0.5)

        # Parallelize the model
        model = multi_gpu_model(model, gpus=None)

        # Set up optimizer
        optimizer = Adam()

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['binary_accuracy'])

        # Train the parallel model using GPUs
        training_history = model.fit(X_train, y_train,
                                     epochs=40,
                                     batch_size=128,
                                     shuffle=True,
                                     validation_split=0.1,
                                     class_weight=class_weight,
                                     verbose=1)

        with tf.device('/cpu:0'):

            # Make predictions and convert labels from one hot to sparse
            y_pred = model.predict(X_test, batch_size=64)

            # Round output from sigmoid activation
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            # Get the classification report
            report = classification_report(y_test,
                                           y_pred,
                                           target_names=['activity',
                                                         'not_activity']
                                           )

            # Add classification report to the dictionary of reports
            reports[i] = report

    # Build a multimodal classifier if requested
    if mode in ['early', 'joint']:

        # Split both images and texts into training and validation sets
        X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
            images, labels,
            test_size=0.10,
            random_state=s)

        X_train_txt, X_val_txt = train_test_split(texts,
                                                  test_size=0.10,
                                                  random_state=s)

        # Use CPU to compile the model
        with tf.device('/cpu:0'):

            # Build early fusion model if requested
            if mode == 'early':

                model = NASNetL_mm.build(neurons=1, l2_penalty=0.000001,
                                         dropout=0.5, activation='sigmoid')

            # Build joint fusion model if requested
            if mode == 'joint':

                model = NASNetL_mm_F.build(l2_penalty=0.000001, dropout=0.5,
                                           neurons=1, activation='sigmoid')

        # Parallelize the model
        model = multi_gpu_model(model, gpus=None)

        # Set up optimizer
        optimizer = Adam()

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['binary_accuracy'])

        # Train the parallel model using GPUs
        training_history = model.fit([X_train_img, X_train_txt], y_train_img,
                                     epochs=40,
                                     batch_size=128,
                                     shuffle=True,
                                     validation_split=0.1,
                                     class_weight=class_weight,
                                     verbose=1)

        with tf.device('/cpu:0'):

            # Make predictions and convert labels from one hot to sparse
            y_pred = model.predict([X_test_img, X_val_txt], batch_size=64)

            # Round output from sigmoid activation
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            # Get the classification report
            report = classification_report(y_test_img,
                                           y_pred,
                                           target_names=['activity',
                                                         'not_activity']
                                           )

            # Add classification report to the dictionary of reports
            reports[i] = report

# Print reports
for k, v in reports.items():
    print("Results for fold {}:\n".format(k))
    print(v)
