from keras.models import Model
from keras.layers import Dense, Input
import keras

ENCODING_DIM_INPUT = 100
ENCODING_DIM_LAYER1 = 128
ENCODING_DIM_LAYER2 = 64
ENCODING_DIM_LAYER3 = 32
ENCODING_DIM_OUTPUT = 16
BATCH_SIZE = 32
EPOCHS = 8


def sae_v1(x_train):

    encode_layer1 = Dense(ENCODING_DIM_LAYER1, activation='relu')
    decode_layer1 = Dense(ENCODING_DIM_INPUT, activation='relu')
    encode_layer2 = Dense(ENCODING_DIM_LAYER2, activation='relu')
    decode_layer2 = Dense(ENCODING_DIM_LAYER1, activation='relu')
    encode_layer3 = Dense(ENCODING_DIM_LAYER3, activation='relu')
    decode_layer3 = Dense(ENCODING_DIM_LAYER2, activation='relu')

    # train layer1
    input_image = Input(shape=(ENCODING_DIM_INPUT, ))
    encode_feature1 = encode_layer1(input_image)
    decode_feature1 = decode_layer1(encode_feature1)
    layer1 = Model(inputs=input_image, outputs=decode_feature1)
    layer1.compile(optimizer='adam', loss='mse')
    layer1.fit(x_train, x_train, epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True)
    feature1 = layer1.predict(x_train)

    # train layer2
    input_feature1 = Input(shape=(ENCODING_DIM_LAYER1, ))
    encode_feature2 = encode_layer2(input_feature1)
    decode_feature2 = decode_layer2(encode_feature2)
    layer2 = Model(inputs=input_feature1, outputs=decode_feature2)
    layer2.compile(optimizer='adam', loss='mse')
    layer2.fit(feature1, feature1, epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True)
    feature2 = layer2.predict(feature1)

    # train layer3
    input_feature2 = Input(shape=(ENCODING_DIM_LAYER2, ))
    encode_feature3 = encode_layer3(input_feature2)
    decode_feature3 = decode_layer3(encode_feature3)
    layer3 = Model(inputs=input_feature2, outputs=decode_feature3)
    layer3.compile(optimizer='adam', loss='mse')
    layer3.fit(feature2, feature2, epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True)

    sae = keras.Sequential()
    sae.add(encode_layer1)
    sae.add(encode_layer2)
    sae.add(encode_layer3)

    return sae


def sae_v2(x_train):

    # input placeholder
    input_image = Input(shape=(ENCODING_DIM_INPUT, ))

    # encoding layer
    encode_layer1 = Dense(ENCODING_DIM_LAYER1, activation='relu')(input_image)
    encode_layer2 = Dense(ENCODING_DIM_LAYER2,
                          activation='relu')(encode_layer1)
    encode_layer3 = Dense(ENCODING_DIM_LAYER3,
                          activation='relu')(encode_layer2)
    encode_output = Dense(ENCODING_DIM_OUTPUT)(encode_layer3)

    # decoding layer
    decode_layer1 = Dense(ENCODING_DIM_LAYER3,
                          activation='relu')(encode_output)
    decode_layer2 = Dense(ENCODING_DIM_LAYER2,
                          activation='relu')(decode_layer1)
    decode_layer3 = Dense(ENCODING_DIM_LAYER1,
                          activation='relu')(decode_layer2)
    decode_output = Dense(ENCODING_DIM_INPUT, activation='tanh')(decode_layer3)

    # build autoencoder, encoder
    autoencoder = Model(inputs=input_image, outputs=decode_output)
    encoder = Model(inputs=input_image, outputs=encode_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, shuffle=True)

    return encoder
