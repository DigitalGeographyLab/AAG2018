# -*- coding: utf-8 -*-

# Import packages
from keras.applications import NASNetLarge
from keras.layers import Dense, Dropout, Input, concatenate, average
from keras.models import Model
from keras.regularizers import l2


class NASNetL:
    @staticmethod
    def build(neurons, l2_penalty, dropout):

        # Initialize the base model
        base_model = NASNetLarge(include_top=False, weights='imagenet',
                                 pooling='avg')

        # Freeze the model
        for layer in base_model.layers:
            layer.trainable = False

        # Build the network head
        head = NASNetL.build_head(base_model, neurons, l2_penalty, dropout)

        # Combine the base model and network head
        model = Model(inputs=base_model.input, outputs=head)

        # Return the model
        return model

    @staticmethod
    def build_head(base_model, neurons, l2_penalty, dropout):

        # Build the network head
        net_head = base_model.output
        net_head = Dropout(dropout)(net_head)
        net_head = Dense(neurons, activation='relu',
                         kernel_regularizer=l2(l2_penalty),
                         activity_regularizer=l2(l2_penalty))(net_head)
        net_head = Dropout(dropout)(net_head)
        net_head = Dense(1, activation='sigmoid')(net_head)

        return net_head


class NASNetL_mm:
    """A large NASNet for multimodal classification with early fusion."""
    @staticmethod
    def build(l2_penalty, dropout, activation='softmax', neurons=2):

        # Initialize and freeze the pre-trained base model
        base_model = NASNetLarge(include_top=False, weights='imagenet',
                                 pooling='avg')

        for layer in base_model.layers:
            layer.trainable = False

        # Add dropout
        image = Dropout(dropout)(base_model.output)

        # Add a dense layer with 300 neurons to match the shape of text input
        image = Dense(300, activation='relu',
                      kernel_regularizer=l2(l2_penalty),
                      activity_regularizer=l2(l2_penalty))(image)

        # Text input
        caption = Input(shape=(300,), dtype='float32', name='caption')

        # Concatenate inputs for both image and text
        network_head = concatenate([image, caption], axis=-1)
        network_head = Dropout(dropout)(network_head)

        # Add a hidden layer with 64 neurons
        network_head = Dense(64, activation='relu',
                             kernel_regularizer=l2(l2_penalty),
                             activity_regularizer=l2(l2_penalty))(network_head)

        # Add dropout
        network_head = Dropout(dropout)(network_head)

        # Add the final layer
        network_head = Dense(neurons, activation=activation,
                             kernel_regularizer=l2(l2_penalty),
                             activity_regularizer=l2(l2_penalty))(network_head)

        # Combine the base model and network head
        model = Model(inputs=[base_model.input, caption], outputs=network_head)

        # Return the model
        return model


class NASNetL_mm_F:
    """A large NASNet for multimodal classification with joint fusion."""
    @staticmethod
    def build(l2_penalty, dropout, activation='softmax', neurons=2):

        # Initialize and freeze the pre-trained base model
        base_model = NASNetLarge(include_top=False, weights='imagenet',
                                 pooling='avg')

        for layer in base_model.layers:
            layer.trainable = False

        # Add dropout
        image = Dropout(dropout)(base_model.output)

        # Add a dense layer with 300 neurons to match the shape of text input
        image = Dense(300, activation='relu',
                      kernel_regularizer=l2(l2_penalty),
                      activity_regularizer=l2(l2_penalty))(image)

        # Define auxiliary input for post captions
        caption = Input(shape=(300,), dtype='float32', name='caption')

        # Build network head; start by averaging the inputs
        network_head = average([image, caption])
        network_head = Dropout(dropout)(network_head)

        # Add a hidden layer with 64 neurons
        network_head = Dense(64, activation='relu',
                             kernel_regularizer=l2(l2_penalty),
                             activity_regularizer=l2(l2_penalty))(network_head)

        # Add dropout
        network_head = Dropout(dropout)(network_head)

        # Add the final layer
        network_head = Dense(neurons, activation=activation,
                             kernel_regularizer=l2(l2_penalty),
                             activity_regularizer=l2(l2_penalty))(network_head)

        # Combine the base model and network head for the final model
        model = Model(inputs=[base_model.input, caption], outputs=network_head)

        # Return the model
        return model


class CaptionClassifier:
    def __init__(self):
        pass

    @staticmethod
    def build(l2_penalty=0.0001, dropout=0.5, activation='softmax', neurons=2):

        # Set up the input layer
        input_layer = Input(shape=(300,), dtype='float32')

        # Add dropout
        model = Dropout(dropout)(input_layer)

        # Add a hidden layer with 64 neurons
        model = Dense(64, activation='relu',
                      kernel_regularizer=l2(l2_penalty),
                      activity_regularizer=l2(l2_penalty))(model)

        # Add dropout
        model = Dropout(dropout)(model)

        # Add the final layer
        model = Dense(neurons, activation=activation,
                      kernel_regularizer=l2(l2_penalty),
                      activity_regularizer=l2(l2_penalty))(model)

        # Wrap the model
        model = Model(inputs=input_layer, outputs=model)

        # Return the model
        return model
