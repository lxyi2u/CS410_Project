from model import lstm, multilayer_lstm, model_cat
from datautil import DataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from log import LossHistory
from keras.models import load_model

INPUT_LENGTH = 50
INPUT_FEATURE = 136
ENCODING_DIM_OUTPUT = 16
BATCH_SIZE = 128
EPOCHS = 8
COUNT = 1721577
WINDOW_LEN = 50
PREDICT_DAYS = 10
TRAIN_NUM = int(COUNT*0.6) - WINDOW_LEN - PREDICT_DAYS + 1
VAL_NUM = int(COUNT*0.1) - WINDOW_LEN - PREDICT_DAYS + 1
TEST_NUM = int(COUNT*0.3) - WINDOW_LEN - PREDICT_DAYS + 1
FILEPATH = "./model/encoderlstm.h5"

batch_size = [64, 128, 256]
hidden_dim = [64, 128, 256, 512]
window_len = [i*10 for i in range(3, 11)]
dropout = [i/10 for i in range(0, 5)]


def train():

    encoder = load_model('./model/encoder.h5')
    rnn = lstm((INPUT_LENGTH, ENCODING_DIM_OUTPUT))
    model = model_cat(
        encoder, rnn, WINDOW_LEN, INPUT_FEATURE, ENCODING_DIM_OUTPUT)
    # model = multilayer_lstm((INPUT_LENGTH, INPUT_FEATURE))
    # model=linear_regression((INTPUT_LENGTH,INPUT_FEATURE))
    model.summary()

    train_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, WINDOW_LEN, BATCH_SIZE, 'train')
    val_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, WINDOW_LEN, BATCH_SIZE, 'validate')
    # model = lstm((INPUT_LENGTH, INPUT_FEATURE))
    # model = multilayer_lstm((INPUT_LENGTH, INPUT_FEATURE))
    # model=linear_regression((INTPUT_LENGTH,INPUT_FEATURE))
    # model.summary()
    model.compile(optimizer='adam', loss='mse')

    checkpoint = ModelCheckpoint(FILEPATH, monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')
    history = LossHistory()
    model.fit_generator(
        train_dataset,
        steps_per_epoch=np.floor(TRAIN_NUM / BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=np.floor(VAL_NUM / BATCH_SIZE),
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint, earlystop, history]
    )
    # model.save(FILEPATH)
    history.loss_plot('epoch', 'encoderlstm')
    history.loss_plot('batch', 'encoderlstm')


def adjust_para():

    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')
    for b in batch_size:
        for w in window_len:
            train_dataset = DataGenerator(
                './dataset/data.csv', PREDICT_DAYS, w, b, 'train')
            val_dataset = DataGenerator(
                './dataset/data.csv', PREDICT_DAYS, WINDOW_LEN, BATCH_SIZE, 'validate')
            for h in hidden_dim:
                for d in dropout:
                    model_path = 'lstm_b{}_w_{}_h_{}_d{}'.format(b, w, h, d)
                    checkpoint = ModelCheckpoint(
                        model_path, monitor='val_loss', verbose=1,
                        save_best_only=True, mode='min')
                    model = lstm((w, INPUT_FEATURE), h, d)
                    model.compile(optimizer='adam', loss='mse')
                    history = LossHistory()
                    model.fit_generator(
                        train_dataset,
                        steps_per_epoch=train_dataset.get_len()/b,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        validation_steps=val_dataset.get_len()/b,
                        shuffle=True,
                        verbose=1,
                        callbacks=[checkpoint, earlystop, history]
                    )
                    history.loss_plot('epoch', model_path)
                    history.loss_plot('batch', model_path)


if __name__ == "__main__":

    train()
