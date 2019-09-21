from utils import load_flick8r_descriptions, load_partly_descriptions
from Inception_model import load_image_files, load_features_from_file, extract_features_inceptionv3
from sequence_utils import create_tokenizer, max_sequence_length, vocabulary_size, load_glove_file, create_embedding_matrix
from model_no_ray import pad_sequences, define_model, data_generator_batch
import h5py
import numpy as np
from numpy import argmax
from keras.models import load_model
import keras.backend as K
from nltk import word_tokenize, translate
import pickle



def id_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return ""
def word_to_id(w, tokenizer):
    for word, index in tokenizer.word_index.items():
        if w == word:
            return index
    return -1
 
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
        word = id_to_word(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text[9:]

def beam_search_predictions(model, image, tokenizer, max_length, beam_index = 3):
    idx2word = {val:index for index, val in tokenizer.word_index.items()}
    word2idx = {index:val for index, val in tokenizer.word_index.items()}
    # print(word2idx)

    start = [word2idx['startseq']]
    # end = [word2idx["<endseq>"]]
    # print(start)

    start_word = [[start, 0.0]]
    # print(start_word)
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image, par_caps], verbose=0)
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    # print('Intermediate caption %s' % intermediate_caption)
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


def bleu_score(model, img_dir,tokenizer,test_features_dict,max_length,beam_size=3):
    
    captions = {}
    with open(img_dir, 'r') as images_path:
        images = images_path.read().strip().split('\n')

    prediction = open('predicted_captions.txt', 'w')

    for count, image in enumerate(images):
        if count%100 == 0 and count is not 0:
            print('Completed creating beam_search_predictions for: {} out of {}'.format(count, len(images)))

        captions[image] = beam_search_predictions(model, test_features_dict[image.split('.')[0]], tokenizer, max_length, beam_size)
        caption =captions[image]
        prediction.write(image + "\t" + str(caption))
        prediction.flush()
    prediction.close()

    captions_path = open('Flickr8k_text/Flickr8k.token.txt', 'r')
    captions_text = captions_path.read().strip().split('\n')
    
    cap_pair = {}
    for i in captions_text:
        i = i.split("\t")
        i[0] = i[0][:len(i[0]) - 2]
        try:
            cap_pair[i[0]].append(i[1])
        except:
            cap_pair[i[0]] = [i[1]]
    captions_path.close()

    h = []
    r = []
    for image in images:
        h.append(captions[image])
        r.append(cap_pair[image])

    return translate.bleu_score.corpus_bleu(r, h)


def main():
    # load all data
    description_dict = load_flick8r_descriptions()
    train_files, test_files = load_image_files()

    # # divide descriptions pool for train and test data
    train_description_dict = load_partly_descriptions(description_dict, train_files)
    test_description_dict = load_partly_descriptions(description_dict, test_files)
    
    # #Extract features 
    # extract_features_inceptionv3(train_files, "encoded_train_images_inceptionv3.p")
    # extract_features_inceptionv3(test_files, "encoded_test_images_inceptionv3.p")

    # # load image features which are already extracted (saved in file)
    train_features_dict = load_features_from_file('encoded_train_images_inceptionv3.p')
    test_features_dict = load_features_from_file('encoded_test_images_inceptionv3.p')
    for key in train_description_dict.keys():
        all_desc = list()
        for d in train_description_dict[key]:
            d = 'startseq ' + d + ' endseq'
            all_desc.append(d)       
        train_description_dict[key] = all_desc
    for key in test_description_dict.keys():
        all_desc = list()
        for d in test_description_dict[key]:
            d = 'startseq ' + d + ' endseq'
            all_desc.append(d)
        test_description_dict[key] = all_desc
    # # prepare tokenizer, vocabulary for word embedding
    tokenizer = create_tokenizer(train_description_dict)



    max_length = max_sequence_length(train_description_dict)
    vocab_size = vocabulary_size(tokenizer)
    embeddings_index = load_glove_file('glove/glove.6B.300d.txt')
    embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index)

    model = define_model(vocab_size, max_length, embedding_matrix)
    print(K.eval(model.optimizer.lr))
    epochs = 100
    steps = len(train_description_dict)
    for i in range(epochs):
        generator = data_generator_batch(train_description_dict, train_features_dict, tokenizer, 512)
        test_generator = data_generator_batch(test_description_dict, test_features_dict, tokenizer, 512)
        model.fit_generator(generator, epochs=1, steps_per_epoch = 748, verbose=1)
        model.save('model' + str(i + 1) + '.h5')
        # model.save_weights('weights_'+ str(i)+'.h5',overwrite=True)

    print(test_features_dict['3385593926_d3e9c21170'])
    caption = beam_search_predictions(model, test_features_dict['3385593926_d3e9c21170'], tokenizer, max_length)
    print(caption)
    
    img_dir='Flickr8k_text/Flickr_8k.testImages.txt'
    print(bleu_score(model, img_dir,tokenizer,test_features_dict,max_length,beam_size=3))

if __name__ == '__main__':
    main()