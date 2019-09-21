import ray
import gc
from image_encoding import *
from sequence_utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import sys
import os
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import add
def create_sequences(tokenizer, description_dict, train_features):
    # deprected
    # this funcion can only run on one core, thus, the performance is slow
    # create sequences of images, input sequences and output words for an image
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in description_dict.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
                max_length = max_sequence_length(train_description_dict)
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)
                # store
                X1.append(train_features[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

@ray.remote
def get_in_out_seqs(tokenizer, key, desc_list, max_length):
    k_in_out_seqs = []
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence

            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
            # store
            k_in_out_seqs.append((key, in_seq, out_seq))
    return k_in_out_seqs

def create_sequences_multicores(max_length, tokenizer, description_dict, features_dict):
    X1, X2, y = [], [], []

    all_k_in_out_seq_ids = []
    epoch_count = 0
    COUNT_PER_EPOCH = 500
    count = 0
    # walk through each image identifier
    for key, desc_list in description_dict.items():
        all_k_in_out_seq_ids.append(get_in_out_seqs.remote(tokenizer, key, desc_list, max_length))
        count += 1
        if count >= COUNT_PER_EPOCH:
            epoch_count += 1
            print("Epoch %d done" % epoch_count)
            all_k_in_out_seqs = ray.get(all_k_in_out_seq_ids)
            for k_in_out_seqs in all_k_in_out_seqs:
                for (key, in_seq, out_seq) in k_in_out_seqs:
                    X1.append(features_dict[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            # reset
            all_k_in_out_seq_ids = []
            count = 0
            gc.collect()

    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)
    return X1, X2, y

# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder = add([se3, fe2])
    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    return model