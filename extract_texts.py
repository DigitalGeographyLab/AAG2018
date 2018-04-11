# -*- coding: utf-8 -*-

# Import packages
from utils import TextProcessor
import argparse
import pandas as pd


# Set up argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-df", "--dataframe", required=True,
                help="Path to the Pandas DataFrame with image data.")

ap.add_argument("-id", "--image_dir", required=True,
                help="Path to the directory with downloaded images.")

ap.add_argument("-of", "--output_file", required=True,
                help="Path to the HDF5 file in which the features are save.")

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
path_to_df = args['dataframe']
image_dir = args['image_dir']
output_file = args['output_file']

# Read the DataFrame
df = pd.read_pickle(path_to_df)

# Remove irrelevant images, which have value 1 in the column "NonRelevant"
df = df[df['NotRelevant'] != 1]

# Drop rows without a caption
df = df[df['text'] != 'None']

# Initialize TextProcessor
tp = TextProcessor()

df['processed_texts'] = df['text'].apply(lambda x: tp.preprocess(x,
                                                                 emojis=False,
                                                                 mentions=False,
                                                                 hashtags=True,
                                                                 trails=True))

# Drop rows without captions
df = df[df['processed_texts'] != '']

# Save DataFrame to disk
df.to_pickle(output_file)