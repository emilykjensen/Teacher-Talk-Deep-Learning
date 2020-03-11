import numpy as np
import pandas as pd

#%% Load word embeddings into a dictionary
# reference article
# https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db

glove_100_file = "/Users/emilyjensen/Dropbox (Emotive Computing)/Emily CETD/Cathlyn_EDM/common-models-original-download/include/glove.6B.100d.txt"

embeddings_dict = {}

# make a dictionary with key = word and value = vector
with open(glove_100_file, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
        

#%% Generate embeddings for teacher utterances

utterance_file = "utterance-data-with-labels-binary.csv"

data = pd.read_csv(utterance_file).set_index(['ObsID','UtteranceID'])
utterances = data['TranscribedUtterance']

# add columns for embeddings
embeddings_cols = ['glove100_' + str(i) for i in range(100)]
to_add = pd.DataFrame(index = data.index, columns = embeddings_cols, data = np.full((len(data),100), np.nan))
data_embeddings = data.join(to_add)

# for each utterance, average embeddings for words in each utterance
def average_embeddings(utterance):
    embeddings = []
    words = utterance.split()
    for word in words:
        try:
            embeddings.append(embeddings_dict[word])
        except:
            print('{} not found, skipping'.format(word))
    return np.average(embeddings, axis = 0)

# for each utterance, choose element-wise max
def element_max_embeddings(utterance):
    embeddings = []
    words = utterance.split()
    for word in words:
        try:
            embeddings.append(embeddings_dict[word])
        except:
            print('{} not found, skipping'.format(word))
    return np.max(embeddings, axis = 0)