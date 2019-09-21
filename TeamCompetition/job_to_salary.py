import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
np.random.seed(0)
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from keras.optimizers import RMSprop
import nltk
nltk.download('punkt')
nltk.download('wordnet')


np.random.seed(1)

def load_glove_file(file='../CaptionGenerator/glove/glove.6B.300d.txt'):
    # load the whole embedding into memory
    word_to_vec_map = dict()
    f = open(file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_to_vec_map[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(word_to_vec_map))
    return word_to_vec_map

def read_csv(filename = 'salary_by_job.csv'):
    job = []
    salary = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            title = row[0].replace(',', '').replace('"', '').replace('\'', '').lower()
            lmtzr = WordNetLemmatizer()
            lemmatized = [lmtzr.lemmatize(word) for word in word_tokenize(title)]
            title = ' '.join(lemmatized)
            job.append(title)
            salary.append(int(row[1].replace(',', '')))

    X = np.asarray(job)
    Y = np.asarray(salary, dtype=int)

    return X, Y


def salary_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim = input_shape[1], activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    model.summary()
    return model
def sentence_to_avg(sentence, word_to_vec_map):
    words = sentence.lower().split()
    avg = np.zeros(300)
    for w in words:
        if w in word_to_vec_map.keys():
            avg += word_to_vec_map[w]
    avg = avg / len(words)   
    return avg

def main():
    X_train, Y_train = read_csv()
    word_to_vec_map = load_glove_file()
    X_train_embedding = np.array([sentence_to_avg(x, word_to_vec_map) for x in X_train])

    model = salary_model(X_train_embedding.shape)
    model.fit(X_train_embedding, Y_train, epochs = 150, batch_size = 32, shuffle = True)
    model.save('model_job_to_salary' + '.h5')
    X1 = np.array(["software developer", 'developer', 'chemist', 'director', 'managing partner'])
    X1_train_embedding = np.array([sentence_to_avg(x, word_to_vec_map) for x in X1])

    print(model.predict(X1_train_embedding))

if __name__ == '__main__':
    main()