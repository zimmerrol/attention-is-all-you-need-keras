import numpy as np
import pickle
import click
from random import shuffle as shf
import os
from utility.language_encoder import LanguageEncoder
import pathlib

@click.command()
@click.option("--input-file", "-i", default="./data/en2de.txt", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-data-folder", "-o", default="./data", required=False, type=click.Path(file_okay=False, dir_okay=True))
@click.option("--shuffle", "-s", required=False, type=click.BOOL, default=True)
@click.option("--vocabulary-size", "-v", type=int, default=5000)
@click.option("--train-test-split", "-s", type=float, default=0.7)
@click.option("--delimiter", "-s", type=str, default="\t")
def cmd(input_file, output_data_folder, shuffle, vocabulary_size, train_test_split, delimiter):
    file_name = os.path.basename(input_file)
    data_name = os.path.splitext(file_name)[0]

    os.makedirs(output_data_folder, exist_ok=True)

    target_language_path = os.path.join(output_data_folder, data_name + ".language.target.pkl")
    source_language_path = os.path.join(output_data_folder, data_name + ".language.source.pkl")
    data_path = os.path.join(output_data_folder, data_name + ".tokens.pkl")

    lines = []
    source = []
    target = []
    print(f"opening '{input_file}'")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    if shuffle: 
        shf(lines)

    for line in lines:
        splits = line.split(delimiter)

        if len(splits) != 2:
            continue

        source.append(splits[0])
        target.append(splits[1])

    train_length = int(train_test_split*len(source))

    source_train = [source[:train_length]]
    source_val = [source[train_length:]]

    target_train = [target[:train_length]]
    target_val = [target[train_length:]]

    source_le = LanguageEncoder(vocabulary_size)
    source_train = source_le.fit_transform(source_train)[0]
    source_val = source_le.transform(source_val)[0]
    source_le.save(source_language_path)

    target_le = LanguageEncoder(vocabulary_size)
    target_train = target_le.fit_transform(target_train)[0]
    target_val = target_le.transform(target_val)[0]
    target_le.save(target_language_path)

    with open(data_path, "wb") as f:
        pickle.dump([(source_train, source_val), (target_train, target_val)], f)

    print("Data processed and saved")
    print("Created languages files at:")
    print("\t" + source_language_path)
    print("\t" + target_language_path)
    print("Saved processed data [(source_train, source_val), (target_train, target_val)] at:")
    print("\t" + data_path)

if __name__ == "__main__":
    cmd()