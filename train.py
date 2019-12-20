from model import *

INPUT_LENGTH=50
INPUT_FEATURE=10

def train():

    model=get_lstm((INPUT_LENGTH,INPUT_FEATURE))
    model.summary()




if __name__ == "__main__":
    train()