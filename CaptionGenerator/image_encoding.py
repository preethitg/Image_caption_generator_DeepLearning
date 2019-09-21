import pickle
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.preprocessing import image

def load_images_vgg16(files):
    # files is the list of image file name in one set (either train or dev or test set)
    model = VGG16()
    images_dict = {}
    for name in files:
        img = image.load_img(flicker8k_dataset + name, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = vgg16_preprocess_input(img)
        images_dict[name.split('.')[0]] = img
    return images_dict

def load_images_inception_v3(files):
    images_dict = {}
    for name in files:
        img = image.load_img(flicker8k_dataset + name, target_size=(299, 299))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = inception_v3_preprocess_input(img)
        images_dict[name.split('.')[0]] = img
    return images_dict

def extract_features_vgg16(files, pickle_file_name):
    # files is the list of image file name in one set (either train or dev or test set)
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    images_dict = load_images_vgg16(files)
    features_dict = {}
    for key, img in images_dict.items():
        feature = model.predict(img, verbose=0)
        features_dict[key] = feature
    with open(pickle_file_name, "wb") as encoded_pickle:
        pickle.dump(features_dict, encoded_pickle)

def load_features_from_file(pickle_file_name):
    return pickle.load(open(pickle_file_name, 'rb'))

def main():
    #extract_features_vgg16(train_files, "encoded_train_images_vgg16.p")
    #extract_features_vgg16(test_files, "encoded_test_images_vgg16.p")
    vgg16_train_features_dict = load_features_from_file('encoded_train_images_vgg16.p')
    vgg16_test_features_dict = load_features_from_file('encoded_test_images_vgg16.p')

if __name__ == '__main__':
    main()