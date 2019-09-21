from utils import *
from image_encoding import *
from sequence_utils import *
from model import *
import ray
import h5py
import numpy as np
from numpy import argmax
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text[9:]




def main():
    # load all data
    description_dict = load_flick8r_descriptions()
    train_files, test_files = load_image_files()

    # divide descriptions pool for train and test data
    train_description_dict = load_partly_descriptions(description_dict, train_files)
    test_description_dict = load_partly_descriptions(description_dict, test_files)

    # load image features which are already extracted (saved in file)
    vgg16_train_features_dict = load_features_from_file('encoded_train_images_vgg16.p')
    vgg16_test_features_dict = load_features_from_file('encoded_test_images_vgg16.p')

    # prepare tokenizer, vocabulary for word embedding
    tokenizer = create_tokenizer(train_description_dict)
    max_length = max_sequence_length(description_dict)
    vocab_size = vocabulary_size(tokenizer)

    # prepare input data for model
    ray.init(redis_max_memory=20000000000, object_store_memory=20000000000)
    X1, X2, y = create_sequences_multicores(max_length, tokenizer, train_description_dict, vgg16_train_features_dict)
    X1test, X2test, ytest = create_sequences_multicores(max_length, tokenizer, test_description_dict, vgg16_test_features_dict)
    print("Dump to file - shutdown ray")
    ray.shutdown()
    # pickle.dump(X1, open("X1.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(X2, open("X2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(y, open("y.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


    # pickle.dump(X1test, open("X1test.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(X2test, open("X2test.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(ytest, open("ytest.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # tempX1 = np.load('X1.npy')
    # tempX2 = np.load('X2.npy')
    # tempy = np.load('y.npy')
    # tempX1test = np.load('X1test.npy')
    # tempX2test = np.load('X2test.npy')
    # tempytest = np.load('ytest.npy')
    # with h5py.File('X1.h5', 'w') as hf:
    #     hf.create_dataset("X1",  data=X1)
    # with h5py.File('X1.h5', 'r') as hf:
    #     tempX1 = hf['X1'][:]

    # with h5py.File('X2.h5', 'w') as hf:
    #     hf.create_dataset("X2",  data=X2)
    # with h5py.File('X2.h5', 'r') as hf:
    #     tempX2 = hf['X2'][:]

    # with h5py.File('y.h5', 'w') as hf:
    #     hf.create_dataset("y",  data=y)
    # with h5py.File('y.h5', 'r') as hf:
    #     tempy = hf['y'][:]

    # with h5py.File('X1test.h5', 'w') as hf:
    #     hf.create_dataset("X1test",  data=X1test)
    # with h5py.File('X1test.h5', 'r') as hf:
    #     tempX1test = hf['X1test'][:]

    # with h5py.File('X2test.h5', 'w') as hf:
    #     hf.create_dataset("X2test",  data=X2test)
    # with h5py.File('X2test.h5', 'r') as hf:
    #     tempX2test = hf['X2test'][:]

    # with h5py.File('ytest.h5', 'w') as hf:
    #     hf.create_dataset("ytest",  data=ytest)
    # with h5py.File('ytest.h5', 'r') as hf:
    #     tempytest = hf['ytest'][:]


    model = define_model(vocab_size, max_length)
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # fit model
    model.fit([X1, X2], y, epochs=10, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
    #sample results
    generate_desc(model, tokenizer, vgg16_test_features_dict['3385593926_d3e9c21170'], max_length)

if __name__ == '__main__':
    main()