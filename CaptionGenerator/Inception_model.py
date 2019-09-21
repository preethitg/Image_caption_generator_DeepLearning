import pickle
import tensorflow as tf
import numpy as np
from time import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.preprocessing import image
from keras.models import Model
from utils import load_image_files
import tqdm

flicker8k_dataset = 'Flicker8k_Dataset/'
# Load the inception v3 model
model = InceptionV3(weights='imagenet')
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(flicker8k_dataset+image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = inception_v3_preprocess_input(x)
    return x

# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    # fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def load_images_inception_v3(files):
    images_dict = {}
    for img in tqdm.tqdm(files):
        images_dict[img.split('.')[0]] = encode(img)
    return images_dict

def extract_features_inceptionv3(files, pickle_file_name):
    images_dict = load_images_inception_v3(files)
    with open(pickle_file_name, "wb") as encoded_pickle:
        pickle.dump(images_dict, encoded_pickle)

def load_features_from_file(pickle_file_name):
    return pickle.load(open(pickle_file_name, 'rb'))

def main():
    
    train_files, test_files = load_image_files()
    
    extract_features_inceptionv3(train_files, "encoded_train_images_inceptionv3.p")
    extract_features_inceptionv3(test_files, "encoded_test_images_inceptionv3.p")

if __name__ == '__main__':
    main()