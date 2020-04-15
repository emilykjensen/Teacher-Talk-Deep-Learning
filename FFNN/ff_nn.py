from __future__ import print_function
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('utterance-data-with-labels-binary.csv')

utterances = data['TranscribedUtterance']
data = data.drop(['IsInstructionalStatement','IsDisciplinaryUtterance','IsDisciplinaryStatement','IsEvaluationFollowupIncluded','IsEvaluationValencePositive','CombinedAuthCogUptake','Uptake','IsInstructionalQuestion', 'IsDisciplinaryQuestion','IsStudentResponsePresent','IsSerialQuestion'], axis=1)

respList = np.array([list(data.iloc[0,4:])])
for i in range(1,len(data)):
    respList = np.append(respList, [list(data.iloc[i,4:])], axis=0)
print(respList.shape)

docs = list(utterances.values)


# vectorize bag of words
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(docs)
docs2 = vectorizer.transform(docs).toarray()

X_train, X_test, y_train, y_test = train_test_split(docs2, respList, test_size=0.2)

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
model.fit(X_train, y_train, batch_size=batchsz, epochs=20, validation_data=(X_test, y_test), shuffle=True)
score, acc = model.evaluate(X_test, y_test,batch_size=batchsz)
print('Test score:', score)
print('Test accuracy:', acc)

y_pred = model.predict(X_test)
print('Test AUC:', roc_auc_score(y_test, y_pred, multi_class='ovr'))


##############################################



docs2 = []
for i in range(len(docs)):
     docs2.append([docs[i]])

docs2 = np.asarray(docs2)
docs2.shape

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(docs)
docs2 = vectorizer.transform(docs).toarray()



enc_docs = np.asarray(encoded_docs)
enc_docs2 = []
for i in range(len(enc_docs)):
     enc_docs2.append([enc_docs[i]])

enc_docs2 = np.asarray(enc_docs2)
enc_docs2.shape

X_train, X_test, y_train, y_test = train_test_split(docs2, respList, test_size=0.2)

input_dim = X_train.shape[1]

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batchsz, epochs=2, validation_data=(X_test, y_test), shuffle=True)


enc_docs = np.asarray(encoded_docs)

model.add(Embedding(max_features, 64, input_length=1))
