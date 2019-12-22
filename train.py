from model import lstm
from datautil import DataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
INPUT_LENGTH = 50
INPUT_FEATURE = 136
BATCH_SIZE = 64
EPOCHS = 8
COUNT = 1721577
WINDOW_LEN = 50
PREDICT_DAYS = 10
TRAIN_NUM = int(COUNT*0.7) - WINDOW_LEN - PREDICT_DAYS + 1
TEST_NUM = int(COUNT*0.3) - WINDOW_LEN - PREDICT_DAYS + 1
FILEPATH = "./model/lstm.h5"


def train():

    train_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, WINDOW_LEN, BATCH_SIZE, 'train')
    test_dataset = DataGenerator(
        './dataset/data.csv', PREDICT_DAYS, WINDOW_LEN, BATCH_SIZE, 'test')
    model = lstm((INPUT_LENGTH, INPUT_FEATURE))
    # model=linear_regression((INTPUT_LENGTH,INPUT_FEATURE))
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    checkpoint = ModelCheckpoint(FILEPATH, monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')

    model.fit_generator(
        train_dataset,
        steps_per_epoch=np.ceil(TRAIN_NUM / BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=test_dataset,
        validation_steps=np.ceil(TEST_NUM / BATCH_SIZE),
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint]
    )


if __name__ == "__main__":

    train()
