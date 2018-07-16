import numpy as np
import utility.coco as coco
import h5py

def load_validation_data(maximum_caption_length):
    coco.set_data_dir("./data/coco")
    coco.maybe_download_and_extract()

    _, _, captions_val_raw = coco.load_records(train=False)

    h5 = h5py.File("image.features.val.VGG19.block5_conv4.h5", "r")
    get_data = lambda i: h5[i]

    return captions_val_raw, get_data


def load_training_data(maximum_caption_length):
    coco.set_data_dir("./data/coco")
    coco.maybe_download_and_extract()

    _, _, captions_train_raw = coco.load_records(train=True)
    
    h5 = h5py.File("image.features.train.VGG19.block5_conv4.h5", "r")
    get_data = lambda i: h5[i]

    return captions_train_raw, get_data


def create_vocabulary(maximum_size, text_sets):
    words = dict()
    for texts in text_sets:
        for text in texts:
            for word in text.lower().split():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
    
    words = [item[0] for item in reversed(sorted(words.items(), key=lambda y: y[1]))]
    words = ["<NULL>", "<START>", "<STOP>"] + words
    words = words[:maximum_size]

    word_index_map = {}
    index_word_map = {}
    for i, word in enumerate(words):
        word_index_map[word] = i
        index_word_map[i] = word

    return word_index_map, index_word_map

def encode_text_sets(text_sets, word_index_map):
    encoded_text_sets = []
    for i, texts in enumerate(text_sets):
        encoded_texts = []
        for j, text in enumerate(texts):
            encoded_text = []
            for word in text.split():
                if word.lower() in word_index_map:
                    encoded_text.append(word_index_map[word.lower()])
                else:
                    encoded_text.append(word_index_map["<NULL>"])

            encoded_texts.append(encoded_text)
        encoded_text_sets.append(encoded_texts)

    return encoded_text_sets