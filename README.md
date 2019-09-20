# README #

## CAPTION GENERATOR ##
### Flickr 8r dataset ###

Please download the image dataset and description for training here:  
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip  
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip  


### Glove word vectors ###

Please download the glove word vectors for training here: 
https://github.com/stanfordnlp/GloVe
under Download pre-trained word vectors: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download)
http://nlp.stanford.edu/data/wordvecs/glove.6B.zip  

The dataset is up to 1GB, so we only keep the copy of it on the local machine. Extract the zip file and place the glove folder under source folder. Make sure glove.txt are accesable under the location : 'glove/glove.6B.300d.txt'. This will allow code to read the glove pretrained word vector files.

### Conda - environment set up ###

I highly recommend you to set up virtual environment for this project, that will help you keep the environment clean and not conflicted with other packages required in other projects.   
Tutorial: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html   
 

### Some packages needed for the project
- conda install pillow
- conda install -c conda-forge pydotplus
- Install ray https://ray.readthedocs.io/en/latest/installation.html (might not need to install ray if you don't re-run the loading input for models)
- Install graphviz (Ubuntu): sudo apt-get install graphviz libgraphviz-dev (optional)
- Install graphviz (MAC) brew install graphviz (optional)


### How to run the prototype scripts on your machine:
- Pull the latest code
- The needed script files are: utils.py, image_encoding.py, model.py, sequence_utils.py, caption_generation.py, model_no_ray.py, caption_generation_no_ray.py
- Train the network and compute the output for sample image: python caption_generation_no_ray.py

### utils.py   
This script has the utils function to load the raw data sets, do some pre-processing data.    
### image_encoding.py
This script has the functions related to image encoding using vgg16.   
### sequence_utils.py   
This script has the functions to extract the bag of words from the Flicker descriptions data, build the tokenizer. Pretrained Glove word vectors are used to extract the embedding matrix, that is given as weights for the Keras Embedding Layer in the neural network.
### model.py or model_no_ray.py   
This script defines the model, loading and aggregating the needed inputs for training the model. model_no_ray.py does not have the function to load input data from scratch.   
### caption_generation.py or caption_generation_no_ray.py   
This script has the main flow of the application, trains the neural network, does the argmax search to output the caption.    

### Must read:
https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8   
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/    
These tutorial has the implementation for your tasks.    

## TEAM COMPETITION #

### Application ###
Run: python salary_prediction.py   
Enter the name you want to predict salary when you are asked to enter given name and family name   
