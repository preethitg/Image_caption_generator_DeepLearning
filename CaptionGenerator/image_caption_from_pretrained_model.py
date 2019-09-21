from PIL import Image
from utils import load_flick8r_descriptions, load_partly_descriptions
from Inception_model import load_image_files, load_features_from_file, extract_features_inceptionv3
from sequence_utils import create_tokenizer, max_sequence_length, vocabulary_size, load_glove_file, create_embedding_matrix
from model_no_ray import pad_sequences, define_model, data_generator_batch
import h5py
import numpy as np
from numpy import argmax
from keras.models import load_model
import keras.backend as K
from caption_generation_no_ray import beam_search_predictions, generate_desc, bleu_score
from collections import namedtuple
import random, csv

def main():
    # load all data
    description_dict = load_flick8r_descriptions()
    train_files, test_files = load_image_files()

    # # divide descriptions pool for train and test data
    train_description_dict = load_partly_descriptions(description_dict, train_files)
    test_description_dict = load_partly_descriptions(description_dict, test_files)
    
    # #Extract features 
    #extract_features_inceptionv3(train_files, "encoded_train_images_inceptionv3.p")
    #extract_features_inceptionv3(test_files, "encoded_test_images_inceptionv3.p")

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
    model = load_model('save_model/model96.h5')
    
    Prediction = namedtuple("Prediction", ["Image", "Caption"])
    Result = namedtuple("Result", ["Beam_Size", "Bleu_Score", "Predictions"])

    testing_images={}
    for i in range(5):
        image, value = random.choice(list(test_features_dict.items()))
        testing_images[image] = value

    with open('output.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(('Beam_Size', 'Bleu_Score', 'Predictions'))    # field header    
    
        for beam_index in [3, 5, 7]:
            print('Creating results for beam_index: {} out of: [3, 5, 7]'.format(beam_index))
            Predictions = []
            for image, value in testing_images.items():
                print('Creating beam_search_predictions for image: {}'.format(image))
                caption = beam_search_predictions(model, value, tokenizer, max_length, beam_index)
                Predictions.append(Prediction(Image=image, Caption=caption))

            img_dir='Flickr8k_text/Flickr_8k.testImages.txt'
            print('Calculating bleu_score for beam_index: {} out of: [3, 5, 7]'.format(beam_index))
            Bleu_Score = bleu_score(model, img_dir,tokenizer,test_features_dict,max_length,beam_index)
            
            result = Result(Beam_Size=beam_index, Bleu_Score=Bleu_Score, Predictions=Predictions)
            w.writerow(result)
        

    

    


if __name__ == '__main__':
    main()