import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM, Reshape


def lstm(input_shape, dropout=0):
    '''
        input_shape=(input_length,input_feature)
    '''

    input_length, input_feature = input_shape
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_feature),
                   return_sequences=False, dropout=dropout,
                   recurrent_dropout=dropout))  # TODO: input_shape=(timesteps ,data_dim)
    model.add(Dense(1))

    return model


def multilayer_lstm(input_shape, dropout):

    input_length, input_feature = input_shape
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_feature),
                   return_sequences=False, dropout=dropout,
                   recurrent_dropout=dropout))  # TODO: input_shape=(timesteps ,data_dim)
    model.add(LSTM(128, input_shape=(input_length, input_feature),
                   return_sequences=False, dropout=dropout,
                   recurrent_dropout=dropout))  # TODO: input_shape=(timesteps ,data_dim)
    model.add(LSTM(128, input_shape=(input_length, input_feature),
                   return_sequences=False, dropout=dropout,
                   recurrent_dropout=dropout))  # TODO: input_shape=(timesteps ,data_dim)
    model.add(Dense(1))

    return model


def linear_regression(input_shape):

    input_length, input_feature = input_shape

    model = Sequential()
    model.add(Reshape((-1,)), input_shape=((input_length, input_feature)))
    model.add(Dense(1))

    return model


def nerual_network(input_shape):

    input_length, input_feature = input_shape
    model = Sequential()
    model.add(Reshape((-1,)), input_shape=((input_length, input_feature)))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model
