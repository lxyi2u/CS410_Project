import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM, RepeatVector, TimeDistributed


def get_lstm(input_shape):

    '''
        input_shape=(input_length,input_feature)
    '''
    
    input_length,input_feature=input_shape
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length,input_feature), return_sequences=False))  # TODO: input_shape=(timesteps ,data_dim)
    model.add(Dense(1))
    
    return model
    