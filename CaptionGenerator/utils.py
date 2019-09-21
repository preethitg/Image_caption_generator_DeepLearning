import sys
import os
import numpy as np
import pandas as pd
import glob

__all__ = ["preprocess", "load_flick8r_descriptions", "load_image_files", 'load_partly_descriptions']

def preprocess(t):
    ret = []
    for w in t.split(' '):
        if not w:
            continue
        # remove punctuation, number and special chars (e.g., #)
        # remove hanging word 's', 'a', 'b'
        w0 = ''.join(c for c in w if c.isalpha() or len(c) > 1)
        # convert to lower case
        ret.append(w0.lower())
        # return the description string with trailing spaces removed
    return (' '.join(ret)).rstrip()
    
def load_flick8r_descriptions():
    flick8r_token_file = 'Flickr8k_text/Flickr8k.token.txt'
    description_dict = {} 
    with open(flick8r_token_file, 'r') as f:
        token_doc = f.read()
        lines = token_doc.split('\n')
        for line in lines:
            if line:
                # get image id, relevant descriptions
                tokens = line.split()
                image_name = tokens[0].split('.')[0] #only get the file name, do not get the extension part
                image_desc = ' '.join(tokens[1:])
                if image_name not in description_dict:
                    description_dict[image_name] = list()
                # preprocess description
                description_dict[image_name].append(preprocess(image_desc))
    return description_dict

def load_image_files():
    flicker8k_dataset = 'Flicker8k_Dataset/'
    image_files = glob.glob(flicker8k_dataset + '*.jpg')
    # define training, validation, and test set
    # these are the image name list of each set
    original_train_files = open('Flickr8k_text/Flickr_8k.trainImages.txt', 'r').read().strip().split('\n')
    original_validation_files = open('Flickr8k_text/Flickr_8k.devImages.txt', 'r').read().strip().split('\n')
    original_test_files = open('Flickr8k_text/Flickr_8k.testImages.txt', 'r').read().strip().split('\n')
    train_files = original_train_files
    test_files = original_test_files
    return train_files, test_files

def load_partly_descriptions(description_dict, files):
    # files is the list of image file name in one set (either train or dev or test set)
    # load descriptions for train or test set only
    result = {}
    for file in files:
        name = file.split('.')[0]
        result[name] = description_dict[name]
    return result

def main():
    description_dict = load_flick8r_descriptions()
    print(description_dict['997338199_7343367d7f'])
    train_files, test_files = load_image_files()
    print(len(train_files))
    print(train_files[600])
    train_description_dict = load_partly_descriptions(description_dict, train_files)
    test_description_dict = load_partly_descriptions(description_dict, test_files)
    print(train_description_dict)

if __name__ == '__main__':
    main()