# -*- coding: utf-8 -*-

from urllib.parse import urlparse
import emoji
import fasttext
import numpy as np
import os
import re
import string


class TextProcessor:
    def __init__(self):
        pass

    @staticmethod
    def preprocess(row, emojis=False, mentions=False, trails=False,
                   hashtags=False):

        if not emojis:

            # If emojis is false, convert unicode emoji to shortcode emoji
            row = emoji.demojize(row)

            # Remove single emojis and their groups
            row = re.sub(r':(?<=:)([a-zA-Z0-9_\-&\'’]*)(?=:):', '', row)

        if not mentions:

            # If mentions is False, remove all mentions (@) in the caption
            row = re.sub(r'@\S+ *', '', row)

        if not hashtags:

            # If hashtags is False, remove all hashtags (#) in the caption
            row = re.sub(r'#\S+ *', '', row)

        # Split the string into a list
        row = row.split()

        # Remove all non-words such as smileys etc. :-)
        row = [word for word in row if re.sub('\W', '', word)]

        # Check the list for URLs and remove them
        row = [word for word in row if not urlparse(word).scheme]

        # Strip extra linebreaks from list items
        row = [word.rstrip() for word in row]

        if not trails:

            # If trails is False, remove hashtags trailing the text e.g. "This
            # is the caption and here are #my #hashtags"
            while len(row) != 0 and row[-1].startswith('#'):
                row.pop()

            # Reconstruct the row
            row = ' '.join(row)

            # Drop hashes from any remaining hashtags
            row = re.sub(r'g*#', '', row)

        # Finally, check that row is a string
        if type(row) is not str:
            row = ' '.join(row)

        # Load punctuation characters
        exclude_chars = set(string.punctuation)

        # Remove punctuation
        row = ''.join([char for char in row if char not in exclude_chars])

        # Return the preprocessed row
        return row


class WordEmbeddings:
    def __init__(self, model):
        self.model = fasttext.load_model(model, encoding='utf-8')

    def get_vector(self, text):

        # Get sentence vector
        vector = self.model[text]

        # Return vector
        return vector
