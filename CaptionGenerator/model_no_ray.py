from sequence_utils import max_sequence_length
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
from keras.optimizers import RMSprop, Adam
# def create_sequences(tokenizer, description_dict, train_features):
#     # deprected
#     # this funcion can only run on one core, thus, the performance is slow
#     # create sequences of images, input sequences and output words for an image
#     X1, X2, y = list(), list(), list()
#     # walk through each image identifier
#     for key, desc_list in description_dict.items():
#         # walk through each description for the image
#         for desc in desc_list:
#             # encode the sequence
#             seq = tokenizer.texts_to_sequences([desc])[0]
#             # split one sequence into multiple X,y pairs
#             for i in range(1, len(seq)):
#                 # split into input and output pair
#                 in_seq, out_seq = seq[:i], seq[i]
#             # pad input sequence
#                 max_length = max_sequence_length(train_description_dict)
#                 in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#                 # encode output sequence
#                 out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)
#                 # store
#                 X1.append(train_features[key][0])
#                 X2.append(in_seq)
#                 y.append(out_seq)
#     return np.array(X1), np.array(X2), np.array(y)

def data_generator_batch(description_dict, train_features, tokenizer, batch_size=128):
    count = 0
    in_img, in_seq, out_word = list(), list(), list()
    while 1:
        for key, desc_list in description_dict.items():
            photo = train_features[key][0]
            max_length = max_sequence_length(description_dict)
            # in_img, in_seq, out_word = create_sequences_progressive_loading(tokenizer, desc_list, photo, max_length)
            for desc in desc_list:
                count += 1
                # encode the sequence
                seq = tokenizer.texts_to_sequences([desc])[0]
                # split one sequence into multiple X,y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_s, out_s = seq[:i], seq[i]
                    # pad input sequence

                    # in_s = pad_sequences([in_s], maxlen=max_length)[0]
                    # encode output sequence
                    out_s = to_categorical([out_s], num_classes=len(tokenizer.word_index) + 1)[0]
                    # store
                    in_img.append(photo)
                    in_seq.append(in_s)
                    out_word.append(out_s)
            if count >= batch_size:
                X1 = np.array(in_img)
                in_seq = pad_sequences(in_seq, maxlen=max_length, padding='post')
                X2 = np.array(in_seq)
                y = np.array(out_word)
                yield [[X1, X2], y]
                in_img, in_seq, out_word = list(), list(), list()


def data_generator(description_dict, train_features, tokenizer):
    while 1:
        for key, desc_list in description_dict.items():
            photo = train_features[key][0]
            max_length = max_sequence_length(description_dict)
            in_img, in_seq, out_word = create_sequences_progressive_loading(tokenizer, desc_list, photo, max_length)
            yield [[in_img, in_seq], out_word]

def create_sequences_progressive_loading(tokenizer, desc_list, photo, max_length):
    X1, X2, y = list(), list(), list()
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
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# define the captioning model
def define_model(vocab_size, max_length, embedding_matrix):
    # feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 300, mask_zero=True, weights=[embedding_matrix], trainable=False)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder = add([se3, fe2])
    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    # summarize model
    print(model.summary())
    return model
