import numpy as np
import pandas as pd

#%% Load word embeddings into a dictionary
# reference article
# https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db


# you can find this file at https://nlp.stanford.edu/projects/glove
glove_100_file = "/path/to/glove.6B.100d.txt"

embeddings_dict = {}

# make a dictionary with key = word and value = vector
with open(glove_100_file, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
        

#%% Generate embeddings for teacher utterances
# Implementation notes: 
    #everything to lowercase
    #skip unknown words, which impacts averaging
    #if all words are unknown, return array of 0's

# this file is not shared publicly
utterance_file = "utterance-data-with-labels-binary.csv"

data = pd.read_csv(utterance_file).set_index(['ObsID','UtteranceID'])
utterances = data['TranscribedUtterance']

# for each utterance, average embeddings for words in each utterance
def average_embeddings(utterance):
    embeddings = []
    words = utterance.lower().split()
    for word in words:
        try:
            embeddings.append(embeddings_dict[word])
        except:
            print('skipped: {}'.format(word))
    return np.average(embeddings, axis = 0)

# for each utterance, choose element-wise max
def element_max_embeddings(utterance):
    embeddings = []
    words = utterance.lower().split()
    for word in words:
        try:
            embeddings.append(embeddings_dict[word])
        except:
            print('skipped: {}'.format(word))
    if len(embeddings) == 0:
        # this happens when all of the words in an utterance are unknown
        return np.zeros((100,))
    else:
        return np.max(embeddings, axis = 0)

# generate embeddings and combine with labels
data['avg_embeddings'] = utterances.apply(average_embeddings)
data = data.join(data['avg_embeddings'].apply(pd.Series).rename(columns = lambda x : 'glove100avg_' + str(x)))

data['max_embeddings'] = utterances.apply(element_max_embeddings)
data = data.join(data['max_embeddings'].apply(pd.Series).rename(columns = lambda x : 'glove100max_' + str(x)))

# this file is not shared publicly
data.to_csv('utterance-data-with-labels-binary-glove100-embeddings-draft.csv')
