from model import lstm, multilayer_lstm, model_cat
from datautil import DataGenerator, DataCertainIntervalGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from log import LossHistory
from keras.models import load_model

INPUT_LENGTH = 50
INPUT_FEATURE = 136
ENCODING_DIM_OUTPUT = 128
BATCH_SIZE = 128
EPOCHS = 32
COUNT = 1721577
WINDOW_LEN = 50
PREDICT_DAYS = 10
TRAIN_NUM = int(COUNT*0.6) - WINDOW_LEN - PREDICT_DAYS + 1
VAL_NUM = int(COUNT*0.1) - WINDOW_LEN - PREDICT_DAYS + 1
TEST_NUM = int(COUNT*0.3) - WINDOW_LEN - PREDICT_DAYS + 1
LOGNAME = 'sae_sigmoid_512_lstm'
FILEPATH = "./model/" + LOGNAME
batch_size = [128, 256, 512, 1024, 2048]
hidden_dim = [64, 128, 256, 512]
window_len = [i*10 for i in range(3, 11)]
dropout = [i/10 for i in range(0, 5)]


def train():

    # encoder = load_model('./model/encoder.h5')
    sae = load_model('./model/sae_sigmoid_512.h5')
    rnn = lstm((INPUT_LENGTH, ENCODING_DIM_OUTPUT))
    model = model_cat(
        sae, rnn, WINDOW_LEN, INPUT_FEATURE, ENCODING_DIM_OUTPUT)
    # model = multilayer_lstm((INPUT_LENGTH, INPUT_FEATURE))
    # model=linear_regression((INTPUT_LENGTH,INPUT_FEATURE))
    model.summary()

    train_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, WINDOW_LEN, BATCH_SIZE, 'train')
    print('steps_per_epoch:', train_dataset.get_len())
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
        steps_per_epoch=train_dataset.get_len(),
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=val_dataset.get_len(),
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint, earlystop, history]
    )
    # model.save(FILEPATH)
    history.loss_plot('epoch', LOGNAME)
    history.loss_plot('batch', LOGNAME)


def adjust_para():

    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')
    # for b in batch_size:
    #      for w in window_len:
    b = 128
    w = 50
    train_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, w, b, 'train')
    val_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, w, b, 'validate')
    for h in hidden_dim:
        for d in dropout:
            print('batch_size:{},window_len:{},hidden_dim:{} \
                dropout:{}'.format(b, w, h, d))
            model_name = 'lstm_b{}_w_{}_h_{}_d{}'.format(
                b, w, h, d)
            model_path = './model/'+model_name
            checkpoint = ModelCheckpoint(
                model_path, monitor='val_loss', verbose=1,
                save_best_only=True, mode='min')
            model = lstm((w, INPUT_FEATURE), h, d)
            model.compile(optimizer='adam', loss='mse')
            history = LossHistory()
            model.fit_generator(
                train_dataset,
                steps_per_epoch=train_dataset.get_len(),
                epochs=EPOCHS,
                validation_data=val_dataset,
                validation_steps=val_dataset.get_len(),
                shuffle=True,
                verbose=1,
                callbacks=[checkpoint, earlystop, history]
            )
            history.loss_plot('epoch', model_name)
            history.loss_plot('batch', model_name)
    '''
    for b in batch_size:
        for w in window_len:
            train_dataset = DataGenerator(
                './dataset/data.csv', PREDICT_DAYS, w, b, 'train')
            val_dataset = DataGenerator(
                './dataset/data.csv', PREDICT_DAYS, w, b, 'validate')
            for h in hidden_dim:
                for d in dropout:
                    model_path = './model/mutilstm_b{}_w_{}_h_{}_d{}' \
                        .format(b, w, h, d)
                    checkpoint = ModelCheckpoint(
                        model_path, monitor='val_loss', verbose=1,
                        save_best_only=True, mode='min')
                    model = multilayer_lstm((w, INPUT_FEATURE), h, d)
                    model.compile(optimizer='adam', loss='mse')
                    history = LossHistory()
                    model.fit_generator(
                        train_dataset,
                        steps_per_epoch=train_dataset.get_len(),
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        validation_steps=val_dataset.get_len(),
                        shuffle=True,
                        verbose=1,
                        callbacks=[checkpoint, earlystop, history]
                    )
                    history.loss_plot('epoch', model_path)
                    history.loss_plot('batch', model_path)
    '''


def adj_batch_size():
    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')
    w = 50
    h = 128
    d = 0
    l = [64]
    for b in l:
        train_dataset = DataGenerator(
            './dataset/data.csv', PREDICT_DAYS, w, b, 'train')
        val_dataset = DataGenerator(
            './dataset/data.csv', PREDICT_DAYS, w, b, 'validate')
        print('batch_size:{},window_len:{},hidden_dim:{} \
            dropout:{}'.format(b, w, h, d))
        model_name = 'lstm_b{}_w_{}_h_{}_d{}'.format(
            b, w, h, d)
        model_path = './model/' + model_name
        checkpoint = ModelCheckpoint(
            model_path, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min')
        model = lstm((w, INPUT_FEATURE), h, d)
        model.compile(optimizer='adam', loss='mse')
        history = LossHistory()
        model.fit_generator(
            train_dataset,
            steps_per_epoch=train_dataset.get_len(),
            epochs=EPOCHS,
            validation_data=val_dataset,
            validation_steps=val_dataset.get_len(),
            shuffle=True,
            verbose=1,
            callbacks=[checkpoint, earlystop, history]
        )
        history.loss_plot('epoch', model_name)
        history.loss_plot('batch', model_name)


def adjust_window():
    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')
    h = 128
    d = 0
    b = 128
    for w in window_len:
        train_dataset = DataGenerator(
            './dataset/data.csv', PREDICT_DAYS, w, b, 'train')
        val_dataset = DataGenerator(
            './dataset/data.csv', PREDICT_DAYS, w, b, 'validate')
        print('batch_size:{},window_len:{},hidden_dim:{} \
            dropout:{}'.format(b, w, h, d))
        model_name = 'lstm_b{}_w_{}_h_{}_d{}'.format(
            b, w, h, d)
        model_path = './model/' + model_name
        checkpoint = ModelCheckpoint(
            model_path, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min')
        model = lstm((w, INPUT_FEATURE), h, d)
        model.compile(optimizer='adam', loss='mse')
        history = LossHistory()
        model.fit_generator(
            train_dataset,
            steps_per_epoch=train_dataset.get_len(),
            epochs=EPOCHS,
            validation_data=val_dataset,
            validation_steps=val_dataset.get_len(),
            shuffle=True,
            verbose=1,
            callbacks=[checkpoint, earlystop, history]
        )
        history.loss_plot('epoch', model_name)
        history.loss_plot('batch', model_name)


def adj_epoch_size():
    h = 128
    d = 0
    b = 128
    w = 50

    train_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, w, b, 'train')
    val_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, w, b, 'validate')
    adj_train(h, d, b, w, train_dataset, val_dataset)


def adj_train(h, d, b, w, train_dataset, val_dataset, i=1):

    earlystop = EarlyStopping(monitor='val_loss', patience=1,
                              verbose=1, mode='auto')

    print('batch_size:{},window_len:{},hidden_dim:{} \
        dropout:{}'.format(b, w, h, d))
    model_name = 'lstm_b{}_w_{}_h_{}_d{}_i{}'.format(
        b, w, h, int(d*10), i)
    model_path = './model/' + model_name + '.h5'
    checkpoint = ModelCheckpoint(
        model_path, monitor='val_loss', verbose=1,
        save_best_only=True, mode='min')
    model = lstm((w, INPUT_FEATURE), h, d)
    model.compile(optimizer='adam', loss='mse')
    history = LossHistory()
    model.fit_generator(
        train_dataset,
        steps_per_epoch=train_dataset.get_len(),
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=val_dataset.get_len(),
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint, earlystop, history]
    )
    history.loss_plot('epoch', model_name+'.jpg')
    history.loss_plot('batch', model_name+'.jpg')


def adj_interval():

    h = 128
    d = 0
    b = 128
    w = 50
    i = 5

    train_dataset = DataCertainIntervalGenerator(
        './dataset/data.csv', PREDICT_DAYS, w, i, b, 'train')
    val_dataset = DataCertainIntervalGenerator(
        './dataset/data.csv', PREDICT_DAYS, w, i, b, 'validate')
    adj_train(h, d, b, w, train_dataset, val_dataset, i)


if __name__ == "__main__":
    # adj_batch_size()
    # adj_epoch_size()
    # adjust_para()
    # adjust_window()
    # train()
    adj_interval()