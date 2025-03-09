import numpy as np
import pandas as pd
import os
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
import sklearn.metrics
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from scipy.stats import trimboth

if __name__ == '__main__':
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # Print out 8 random entries
    tr_text_list = x_train_df['text'].values.tolist()
    y_rawData_list = np.array(y_train_df['Coarse Label'].values.tolist())

    # print(type(y_rawData_list))


    y_rawData_N = (y_rawData_list == "Key Stage 4-5").astype(int)
 
    
    print(y_rawData_N)
    
    # TODO: Figure out what to store
    vectorizer = CountVectorizer(lowercase=True, analyzer='word', token_pattern=r'[a-zA-Z]*')
    coords_occur_list = vectorizer.fit_transform(tr_text_list)
    word_list = vectorizer.get_feature_names_out(tr_text_list)
    x_rawData_NF = vectorizer.fit_transform(tr_text_list).toarray()

    x_train_df = trimboth(x_train_df, 0.25)
    y_train_df = trimboth(y_train_df, 0.25)
    
    
    N, F = x_rawData_NF.shape
    N1, = y_rawData_N.shape

    assert(N == N1)

    # Preprocessing the data
    scaler = preprocessing.StandardScaler().fit(x_rawData_NF)
    x_rawData_NF = scaler.transform(x_rawData_NF)

    # print(x_rawData_NF.shape)
    # print(y_rawData_list.shape)
    

    # print(coords_occur_list.toarray())
    
    clf = sklearn.linear_model.LogisticRegression(penalty='l2',C=1.0, max_iter=5000, solver='lbfgs')
    print("N", N)
    print("F", F)

    x_tr, x_te, y_tr, y_te = train_test_split(x_rawData_NF, y_rawData_N, test_size=0.2, random_state=69)

    kf = KFold(n_splits=5, shuffle=True, random_state=69, )

    model_per_fold = list() 
    tr_err_list = list()
    te_err_list = list()
    
    for train_idx, test_idx in kf.split(x_tr):
        x_train_split, x_test_split = x_tr[train_idx], x_tr[test_idx]
        y_train_split, y_test_split = y_tr[train_idx], y_tr[test_idx]
        
        model_per_fold.append(clf.fit(x_train_split, y_train_split))
        tr_err_list.append(1 - clf.score(x_train_split, y_train_split))
        te_err_list.append(1 - clf.score(x_test_split, y_test_split))
        



    print("Training Error", tr_err_list)
    print("Test Error", te_err_list)

  

    ##cross validae 6 folds/ 4 train 2 /test/


    
    

    


    '''
    Stuff to look into
        - Vectorizer function cuts off contractions at the ' (e.g. don't -> don)


    Step 1: Creating the BoW Representation
        - Create a list/dict, document_count_vectors, where each entry represents a document's count vector 
        (a vector were each index holds the num of occurances for a word)
        
        - For each document 
            * Call the vectorizer function and store the returned vector in the document_count_vectors

    Step 2: Log Regression
        - Set our dimensions 
            N = num of documents in vectorized
            V = length of the count vector

        - Load the training y_tr_N values

        - Create a numpy array x_tr_NV using the vectorized data (can literally use document_count_vectors )
            * Each row represents a document 
            * Each col represents the count for a word 

        - Perform cross validation w/ multiple folds
            * For each cross validation w/ k folds
                Save the log regression model to some list

            


    
    
    
    '''