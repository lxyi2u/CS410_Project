from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Reshape, Input
from keras.models import Model
from keras import backend as K
from keras.layers.core import Lambda


def preprocess_reshape(x):
    return K.reshape(x, (-1, 136))


def backend_reshape(x):
    return K.reshape(x, (-1, 50, 16))


def model_cat(encoder, rnn, WINDOW_LEN,
              ENCODING_DIM_INPUT, ENCODING_DIM_OUTPUT=16):

    data = Input(shape=(WINDOW_LEN, ENCODING_DIM_INPUT, ))
    x = Lambda(preprocess_reshape, output_shape=(ENCODING_DIM_INPUT,))(data)
    x = encoder(x)
    x = Lambda(backend_reshape, output_shape=(WINDOW_LEN, ENCODING_DIM_OUTPUT))(x)
    out = rnn(x)

    return Model(inputs=data, output=out)


def lstm(input_shape, hidden_dim=128, dropout=0):
    '''
        input_shape=(input_length,input_feature)
    '''

    input_length, input_feature = input_shape
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=(input_length, input_feature),
                   return_sequences=False, dropout=dropout,
                   recurrent_dropout=dropout))
    model.add(Dense(1))

    return model


def multilayer_lstm(input_shape, dropout=0):

    input_length, input_feature = input_shape
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_feature),
                   return_sequences=True, dropout=dropout,
                   recurrent_dropout=dropout))
    model.add(LSTM(128, input_shape=(input_length, input_feature),
                   return_sequences=True, dropout=dropout,
                   recurrent_dropout=dropout))
    model.add(LSTM(128, input_shape=(input_length, input_feature),
                   return_sequences=False, dropout=dropout,
                   recurrent_dropout=dropout))
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

# class lstm_autoencoder():

#     def __init__(self, label_dataset, unlabeled_dataset,
#                       input_shape, dropout=0):

#         self.label_dataset = label_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.input_length, self.input_feature = input_shape
#         self.dropout = dropout
#         self.model = Sequential()
#         self.model.add(
#             LSTM(128, input_shape=(self.input_length, self.input_feature),
#                  return_sequences=False, dropout=self.dropout,
#                  recurrent_dropout=self.dropout))
#         self.model.add(Dense(1))

#         # encoding layer
#         self.encode_layer1 = Dense(128, activation='relu')
#         self.encode_layer2 = Dense(64, activation='relu')
#         self.encode_layer3 = Dense(32, activation='relu')
#         self.encode_output = Dense(16)

#         # decoding layer
#         self.decode_layer1 = Dense(32, activation='relu')
#         self.decode_layer2 = Dense(64, activation='relu')
#         self.decode_layer3 = Dense(128, activation='relu')
#         self.decode_output = Dense(self.input_feature, activation='tanh')

#     def unsupervise_train(self):

#         # input placeholder
#         input_feature = Input(shape=(self.input_feature, ))

#         # encoding layer
#         out = self.encode_layer1(input_feature)
#         out = self.encode_layer2(out)
#         out = self.encode_layer3(out)
#         encode_feature = self.encode_output(out)

#         # decoding layer
#         out = self.decode_layer1(encode_feature)
#         out = self.decode_layer2(out)
#         out = self.decode_layer3(out)
#         decode_feature = self.decode_output(out)

#         autoencoder = Model(inputs=input_feature, outputs=decode_feature)
#         encoder = Model(inputs=input_feature, outputs=encode_feature)

#         # compile autoencoder
#         autoencoder.compile(optimizer='adam', loss='mse')

#         # training
#         autoencoder.fit(
#             self.unlabeled_dataset,
#             steps_per_epoch=np.floor(TRAIN_NUM / BATCH_SIZE)-5,
#             epochs=EPOCHS,
#             shuffle=True,
#             verbose=1,
#             # callbacks=[checkpoint]
#         )
