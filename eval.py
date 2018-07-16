from model import create_model
from utility.utility import load_training_data, load_validation_data
from utility.language_encoder import LanguageEncoder
import numpy as np
from keras.models import load_model
import pathlib
import pickle
import sys

def inference(inference_model, source):
    output_sentence = [target_le.transform_word("<START>")]
    output_sentence_array = np.array([target_le.transform_word("<START>")] + [target_le.transform_word("<NULL>")]*(MAXIMUM_TEXT_LENGTH-1)).reshape(1, -1)
    source = source.reshape(1, -1)

    current_word = None
    for t in range(MAXIMUM_TEXT_LENGTH):
        data = [source, output_sentence_array]

        output = inference_model.predict(data)
        words = np.argmax(output[0], -1)

        current_word = np.argmax(output[0, t])
        output_sentence.append(current_word)
        output_sentence_array[0, t] = current_word

        if current_word == target_le.transform_word("<STOP>"):
            break

    sentence = [target_le._index_word_map[i] for i in output_sentence[1:]]
    return " ".join(sentence) + " ({0})".format(len(output_sentence)-1)

pathlib.Path('./models').mkdir(exist_ok=True) 

MAXIMUM_TEXT_LENGTH = 70

source_le = LanguageEncoder.load("./data/en2de.language.source.pkl")
target_le = LanguageEncoder.load("./data/en2de.language.target.pkl")

with open("data/en2de.tokens.pkl", "rb") as f:
    (_, source_val), (_, target_val) = pickle.load(f)

print("source vocabulary size: " + str(source_le._vocabulary_size))
print("target vocabulary size: " + str(target_le._vocabulary_size))

source_val = [[source_le.transform_word("<START>")] + item + [source_le.transform_word("<STOP>")] for item in source_val]
target_val = [[target_le.transform_word("<START>")] + item + [target_le.transform_word("<STOP>")] for item in target_val]

source_val = [x[:MAXIMUM_TEXT_LENGTH-1] + [source_le.transform_word("<NULL>")]*(MAXIMUM_TEXT_LENGTH-len(x)) for x in source_val]
target_val = [x[:MAXIMUM_TEXT_LENGTH-1] + [target_le.transform_word("<NULL>")]*(MAXIMUM_TEXT_LENGTH-len(x)) for x in target_val]

source_val = np.array(source_val)
target_val = np.array(target_val)

model_id = input("Model ID: ")
model_id = int(model_id)

_, inference_model = create_model(source_le._vocabulary_size, target_le._vocabulary_size, MAXIMUM_TEXT_LENGTH, n=2, d_model=256, h=4)
try:
    inference_model.load_weights(f"./models/aiayn.inference.{model_id}.h5")
except:
    print("Could not load model from: " + f"./models/aiayn.inference.{model_id}.h5")
    sys.exit()

while True:
    max_idx = len(source_val)
    idx = input(f"Enter the sentence index (0-{max_idx}): ")
    idx = int(idx)
    if idx >= len(source_val):
        continue
    
    print("\ttarget: ")
    print("\t\t{0}".format(" ".join([target_le._index_word_map[x] for x in target_val[idx]])))

    print("\toutput:")
    print("\t\t{0}".format(inference(inference_model, source_val[idx])))
    print("")

input("done. <read key>")
