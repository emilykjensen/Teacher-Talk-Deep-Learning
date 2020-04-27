from __future__ import print_function
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping

data = pd.read_csv('utterance-data-with-labels-binary.csv')

utterances = data['TranscribedUtterance']
data = data.drop(['IsInstructionalStatement','IsDisciplinaryUtterance','IsDisciplinaryStatement','IsEvaluationFollowupIncluded','IsEvaluationValencePositive','CombinedAuthCogUptake','Uptake','IsInstructionalQuestion', 'IsDisciplinaryQuestion','IsStudentResponsePresent','IsSerialQuestion'], axis=1)

respList = np.array([list(data.iloc[0,4:])])

for i in range(1,len(data)):
    respList = np.append(respList, [list(data.iloc[i,4:])], axis=0)

docs = list(utterances.values)

# vectorize
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(docs)
docs2 = vectorizer.transform(docs).toarray()

#GloVe text DO THIS INSTEAD OF COUNT VECTORIZER

info = []

X_temp, X_test, y_temp, y_test = train_test_split(docs2, respList, test_size=0.15, random_state=45)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(15/85), random_state=45)
print(X_train.shape, 'train sequences')
print(X_val.shape, 'validation sequences')
print(X_test.shape, 'test sequences')

layers = [1,2,3]
dropout = [0.1,0.2,0.4]

for x in range(3):
    for z in range(3):

        print('-------->>> STARTING: ', layers[x], ' LAYERS with ', dropout[z], ' DROPOUT <<<--------')
        input_dim = X_train.shape[1]

        model = Sequential()
        model.add(Embedding(input_dim, 64, input_length = input_dim)) #, input_length=input_dim))
        model.add(Flatten())
        # USE A TUTORIAL

        if layers[x] == 1:
            model.add(Dense(7, input_dim=64, activation='relu'))

        if layers[x] == 2:
            model.add(Dense(64, input_dim=64, activation='relu'))
            model.add(Dropout(dropout[z]))
            model.add(Dense(7, activation='sigmoid'))

        if layers[x] == 3:
            model.add(Dense(64, input_dim=64, activation='relu'))
            model.add(Dropout(dropout[z]))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout[z]))
            model.add(Dense(7, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.005, patience=5)

        batchsz = 4
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Train...')
        model.fit(X_train, y_train, batch_size=batchsz, epochs=100, validation_data=(X_val, y_val), shuffle=True, callbacks=[es])
        score, acc = model.evaluate(X_val, y_val,batch_size=batchsz)

        y_pred = model.predict(X_val)

        info.append([[str(layers[x]), ' layers w/ ',str(dropout[z]), ' dropout'],[' score:', score],['accuracy:', acc],['AUC:', roc_auc_score(y_val, y_pred, multi_class='ovr')]])

print("")
print("")
print(info)
print("")
print("")
exit()
