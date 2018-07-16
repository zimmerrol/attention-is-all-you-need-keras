from model import create_model
from utility.utility import load_training_data, load_validation_data
from utility.language_encoder import LanguageEncoder
import numpy as np
import keras.backend as K
from keras.models import load_model
import pathlib
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import pickle

MAXIMUM_TEXT_LENGTH = 70

source_le = LanguageEncoder.load("./data/en2de.language.source.pkl")
target_le = LanguageEncoder.load("./data/en2de.language.target.pkl")

with open("data/en2de.tokens.pkl", "rb") as f:
    (source_train, source_val), (target_train, target_val) = pickle.load(f)

print("source vocabulary size: " + str(source_le._vocabulary_size))
print("target vocabulary size: " + str(target_le._vocabulary_size))

source_train = [[source_le.transform_word("<START>")] + item + [source_le.transform_word("<STOP>")] for item in source_train]
target_train = [[target_le.transform_word("<START>")] + item + [target_le.transform_word("<STOP>")] for item in target_train]

source_train = [x[:MAXIMUM_TEXT_LENGTH-1] + [source_le.transform_word("<NULL>")]*(MAXIMUM_TEXT_LENGTH-len(x)) for x in source_train]
target_train = [x[:MAXIMUM_TEXT_LENGTH-1] + [target_le.transform_word("<NULL>")]*(MAXIMUM_TEXT_LENGTH-len(x)) for x in target_train]

source_train = np.array(source_train)
target_train = np.array(target_train)

print("source_train", source_train.shape)
print("target_train", target_train.shape)

training_model, inference_model = create_model(source_le._vocabulary_size, target_le._vocabulary_size, MAXIMUM_TEXT_LENGTH,
                                                n=2, d_model=256, h=4, optimizer=Adam(0.001, 0.9, 0.98, epsilon=1e-9))
training_model.summary()

batch_size = 64

def train(epochs=100):
    pathlib.Path('./models').mkdir(exist_ok=True) 
    tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False)
    history = training_model.fit([source_train, target_train], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[])

    training_model.save(f"./models/aiayn.train.{epochs}.h5")
    inference_model.save(f"./models/aiayn.inference.{epochs}.h5")

    for key in history.history.keys():
        f = plt.figure()
        data = history.history[key]
        plt.plot(data)
    plt.show()

epochs = input("Number of epochs: ")
epochs = int(epochs)
train(epochs=epochs)
input("done. <read key>")
