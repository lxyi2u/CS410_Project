from model import lstm
from datautil import DataGenerator
from keras.callbacks import ModelCheckpoint
INPUT_LENGTH = 50
INPUT_FEATURE = 136
BATCH_SIZE = 64
EPOCHS = 8
FILEPATH = "./model/lstm.h5"


def train():

    train_dataset = DataGenerator(
        './dataset/data.csv', 10, 50, BATCH_SIZE, 'train')
    test_dataset = DataGenerator(
        './dataset/data.csv', 10, 50, BATCH_SIZE, 'test')
    model = lstm((INPUT_LENGTH, INPUT_FEATURE))
    # model=linear_regression((INTPUT_LENGTH,INPUT_FEATURE))
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    checkpoint = ModelCheckpoint(FILEPATH, monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')

    model.fit_generator(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint]
    )


if __name__ == "__main__":

    train()
