# -*- coding: utf-8 -*-

# Import packages
import argparse
import fasttext
import os
import pandas as pd

# Set up argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-df", "--dataframe", required=True,
                help="Path to the Pandas DataFrame with with pre-processed "
                     "captions.")
ap.add_argument("-of", "--output_file", required=True,
                help="Path to the file in which the embeddings are saved.")

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
path_to_df = args['dataframe']
output_file = args['output_file']


# Read the DataFrame
df = pd.read_pickle(path_to_df)

# Extract captions and write them to a temporary file
df['processed_texts'].to_csv('temp.txt', index=False)

# Train 300-dimensional fastText embeddings and save the model
model = fasttext.skipgram(input_file='temp.txt', output=output_file, epoch=100,
                          lr=0.1, dim=300, silent=False, encoding='utf-8')

# Remove the temporary text file
os.remove('temp.txt')
