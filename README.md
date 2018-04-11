# Applying deep learning to multimodal data in social media

This repository contains code associated with the presentation "Applying deep learning to multimodal data in social media", given at the 2018 American Association for Geographers Annual Meeting, New Orleans, Louisiana. The slides for the presentation are available at: https://www.helsinki.fi/~thiippal/presentations/2018-aag.pdf

## The basics

The repository is structured as follows.

| File | Description |
|:---|:---|
|examine_dataframe.py|A convenience script for printing out the contents of a pandas DataFrame.|
|extract_texts.py|Extract and preprocess captions stored in a pandas DataFrame for learning word embeddings.|
|run_experiment.py|Train classifiers with different architectures over a number of random training / development / testing splits.|
|train_caption_classifier.py|Train a classifier using captions only.|
|train_early_fusion_classifier.py|Train a classifier with early fusion of images and captions.|
|train_image_classifier.py|Train a classifier using images only.|
|train_joint_fusion_classifier.py|Train a classifier with joint fusion of images and captions.|
|train_word_embeddings.py|Learn word embeddings from a corpus of pre-processed captions.|

## How to format your data

## Required libraries

The system was developed using the following libraries. Make sure you have installed the version mentioned below.

| Library | Version |
|:---|:---|
|||
|||

## Reference

Feel free to re-use any of the code in this repository. If the code benefits your published research, please consider citing this work using the following reference:

Hiippala, T., Fink C., Heikinheimo, V., Tenkanen, H. and Toivonen, T. (2018) Applying deep learning to multimodal data in social media. Paper presented at the 2018 American Association of Geographers Annual Meeting, April 10-14, New Orleans, Louisiana.
