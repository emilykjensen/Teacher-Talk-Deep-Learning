# See lines with comment "# KFOLD ADDITION"

from sklearn.model_selection import KFold # KFOLD ADDITION
from sklearn.model_selection import GroupKFold # KFOLD ADDITION

import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

data = pd.read_csv('utterance-data-with-labels-binary.csv')

utterances = data['TranscribedUtterance']
data = data.drop(['IsInstructionalStatement','IsDisciplinaryUtterance','IsDisciplinaryStatement','IsEvaluationFollowupIncluded','IsEvaluationValencePositive','CombinedAuthCogUptake','Uptake','IsInstructionalQuestion', 'IsDisciplinaryQuestion','IsStudentResponsePresent','IsSerialQuestion'], axis=1)

respList = np.array([list(data.iloc[0,4:])])
for i in range(1,len(data)):
    respList = np.append(respList, [list(data.iloc[i,4:])], axis=0)
print(respList.shape)

docs = list(utterances.values)
groups = data['ObsID'] # KFOLD ADDITION
group_kfold = GroupKFold(n_splits=5) # KFOLD ADDITION

# vectorize bag of words
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(docs)
docs2 = vectorizer.transform(docs).toarray()
group_kfold.get_n_splits(docs2, respList, groups) # KFOLD ADDITION

score_array = [] # KFOLD ADDITION
acc_array = [] # KFOLD ADDITION
roc_array = [] # KFOLD ADDITION

for train_index, test_index in group_kfold.split(docs2, respList, groups): # KFOLD ADDITION
    print("TRAIN:", train_index, "TEST:", test_index) # KFOLD ADDITION
    X_train, X_test = docs2[train_index], docs2[test_index] # KFOLD ADDITION
    y_train, y_test = respList[train_index], respList[test_index] # KFOLD ADDITION
    print(X_train, X_test, y_train, y_test) # KFOLD ADDITION

    # X_train, X_test, y_train, y_test = train_test_split(docs2, respList, test_size=0.2)

    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='sigmoid'))

    batchsz = 4
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    print('Train...')
    model.fit(X_train, y_train, batch_size=batchsz, epochs=5, validation_data=(X_test, y_test), shuffle=True)
    score, acc = model.evaluate(X_test, y_test,batch_size=batchsz)
    print('Test score:', score)
    print('Test accuracy:', acc)

    y_pred = model.predict(X_test)
    print('Test AUC:', roc_auc_score(y_test, y_pred, multi_class='ovr'))

    score_array.append(score) # KFOLD ADDITION
    acc_array.append(acc) # KFOLD ADDITION
    roc_array.append(roc_auc_score(y_test, y_pred, multi_class='ovr')) # KFOLD ADDITION

# KFOLD ADDITION
print("Averages Across ", group_kfold.get_n_splits(docs2, respList, groups), " KFold splits is:")
print('Test score:', average(score_array))
print('Test accuracy:', average(acc_array))
print('Test AUC:',average(roc_array))
