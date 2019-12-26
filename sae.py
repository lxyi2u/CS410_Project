from keras.models import Model
from keras.layers import Dense, Input
import keras
from datautil import IdentityDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from log import LossHistory
import numpy as np

ENCODING_DIM_INPUT = 136
ENCODING_DIM_LAYER1 = 128
ENCODING_DIM_LAYER2 = 64
ENCODING_DIM_LAYER3 = 32
ENCODING_DIM_OUTPUT = 16
BATCH_SIZE = 256
EPOCHS = 8


def sae_v1():

    x_train = IdentityDataGenerator(
        './dataset/data.csv',  BATCH_SIZE, 'train')
    x_val = IdentityDataGenerator(
        './dataset/data.csv',  BATCH_SIZE, 'validate')
    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')

    encode_layer1 = Dense(ENCODING_DIM_LAYER1, activation='relu')
    decode_layer1 = Dense(ENCODING_DIM_INPUT, activation='relu')
    encode_layer2 = Dense(ENCODING_DIM_LAYER2, activation='relu')
    decode_layer2 = Dense(ENCODING_DIM_LAYER1, activation='relu')
    encode_layer3 = Dense(ENCODING_DIM_LAYER3, activation='relu')
    decode_layer3 = Dense(ENCODING_DIM_LAYER2, activation='relu')

    # train layer1
    input_data = Input(shape=(ENCODING_DIM_INPUT, ))
    encode_feature1 = encode_layer1(input_data)
    decode_feature1 = decode_layer1(encode_feature1)
    layer1 = Model(inputs=input_image, outputs=decode_feature1)
    layer1.compile(optimizer='adam', loss='mse')
    layer1.fit(
        x_train,
        x_train,
        epochs=1,
        batch_size=BATCH_SIZE,
        shuffer=True
    )
    layer1.save('./model/layer1')
    encode_layer1.trainable = False

    # train layer2
    x_feature1 = layer1.predict(x_train)
    print(x_feature1.shape)
    input_data = Input(shape=(ENCODING_DIM_LAYER1, ))
    encode_feature2 = encode_layer2(input_data)
    decode_feature2 = decode_layer2(encode_feature2)
    layer2 = Model(inputs=input_data, outputs=decode_feature2)
    layer2.summary()
    layer2.compile(optimizer='adam', loss='mse')
    layer2.fit(
        x_feature1,
        x_feature1,
        epochs=1,
        batch_size=BATCH_SIZE,
        shuffer=True
    )
    layer2.save('./model/layer2')
    encode_layer2.trainable = False

    # train layer3
    x_feature2 = layer2.predict(x_feature1)
    print(x_feature2.shape)
    input_data = Input(shape=(ENCODING_DIM_LAYER2, ))
    encode_feature3 = encode_layer2(input_data)
    decode_feature3 = decode_layer2(encode_feature3)
    layer3 = Model(inputs=input_data, outputs=decode_feature3)
    layer3.summary()
    layer3.compile(optimizer='adam', loss='mse')
    layer3.fit(
        x_feature2,
        x_feature2,
        epochs=1,
        batch_size=BATCH_SIZE,
        shuffer=True
    )
    layer3.save('./model/layer2')


    encode_layer1.trainable = True
    encode_layer2.trainable = True

    sae = Model(inputs=input_image, output=encode_feature3)
    sae.save('./model/sae.h5')


def sae_v2():

    # input placeholder
    input_image = Input(shape=(ENCODING_DIM_INPUT, ))

    # encoding layer
    encode_layer1 = Dense(ENCODING_DIM_LAYER1,
                          activation='relu')(input_image)
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
    decode_output = Dense(ENCODING_DIM_INPUT,
                          activation='tanh')(decode_layer3)

    # build autoencoder, encoder
    autoencoder = Model(inputs=input_image, outputs=decode_output)
    encoder = Model(inputs=input_image, outputs=encode_output)

    checkpoint = ModelCheckpoint('./model/autoencoder.h5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')
    history = LossHistory()
    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    train_dataset = IdentityDataGenerator(
        './dataset/data.csv',  BATCH_SIZE, 'train')
    val_dataset = IdentityDataGenerator(
        './dataset/data.csv',  BATCH_SIZE, 'validate')

    autoencoder.fit_generator(
        train_dataset,
        steps_per_epoch=np.floor(train_dataset.get_len() / BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=np.floor(val_dataset.get_len() / BATCH_SIZE),
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint, earlystop, history]
    )

    history.loss_plot('epoch', 'autoencoder')
    history.loss_plot('batch', 'autoencoder')

    encoder.save('./model/encoder.h5')
    autoencoder.save('./model/autoencoder.h5')
    return encoder


if __name__ == "__main__":

    sae_v1()
