# reference article
# https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db

import numpy as np

glove_100_file = "/Users/emilyjensen/Dropbox (Emotive Computing)/Emily CETD/Cathlyn_EDM/common-models-original-download/include/glove.6B.100d.txt"

embeddings_dict = {}

# make a dictionary with key = word and value = vector
with open(glove_100_file, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
        
