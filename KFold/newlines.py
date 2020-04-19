# import these:
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold # StratifiedGroupKFold could be used in the future

# after data processing, add these:
groups = data['ObsID'] # selects column to group by
group_kfold = GroupKFold(n_splits=5) # set number of splits
group_kfold.get_n_splits(docs2, respList, groups) # split where docs2 = utterance values, respList = 7 classifier columns

# loop through each of the 5 splits:

score_array = [] 
acc_array = []
roc_array = []

for train_index, test_index in group_kfold.split(docs2, respList, groups): 
    print("TRAIN:", train_index, "TEST:", test_index) 
    X_train, X_test = docs2[train_index], docs2[test_index]
    y_train, y_test = respList[train_index], respList[test_index] 
    print(X_train, X_test, y_train, y_test) 
    
    # Remove your prior split ie X_train, X_test, y_train, y_test = train_test_split(docs2, respList, test_size=0.2)
    # Run prior training as before
    
    # Append new values in each loop
    score_array.append(score) # KFOLD ADDITION
    acc_array.append(acc) # KFOLD ADDITION
    roc_array.append(roc_auc_score(y_test, y_pred, multi_class='ovr')) # KFOLD ADDITION

# See results
print("Averages Across ", group_kfold.get_n_splits(docs2, respList, groups), " KFold splits is:")
print('Test score:', sum(score_array)/len(score_array))
print('Test accuracy:', sum(acc_array)/len(acc_array))
print('Test AUC:',sum(roc_array)/len(roc_array))
    
