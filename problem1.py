import numpy as np
import pandas as pd
import os
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
import sklearn.metrics

if __name__ == '__main__':
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # Print out 8 random entries
    tr_text_list = x_train_df['text'].values.tolist()
    prng = np.random.RandomState(101)
    rows = prng.permutation(np.arange(y_train_df.shape[0]))
    
    
    # ##print
    # for row_id in rows[:]:
    #     text = tr_text_list[row_id]
    #     print("row %5d | %s BY %s | y = %s" % (
    #         row_id,
    #         y_train_df['title'].values[row_id],
    #         y_train_df['author'].values[row_id],
    #         y_train_df['Coarse Label'].values[row_id],
    #         ))
    #     # Pretty print text via textwrap library
    #     line_list = textwrap.wrap(tr_text_list[row_id],
    #         width=70,
    #         initial_indent='  ',
    #         subsequent_indent='  ')
    #     print('\n'.join(line_list))
    #     print("")
    # tr_text_list = ["I dont know what i am doing know don't",
                #   "Hi this is just easy for don't know doing hi"
                #   ] 
    
    vectorizer = CountVectorizer(lowercase=True, analyzer='word')
    coords_occur_list = vectorizer.fit_transform(tr_text_list)

    
    word_list = vectorizer.get_feature_names_out(tr_text_list)
    # for x in word_list[:]:
    #     index_of_word = vectorizer.vocabulary_.get(x)
    #     print("occurance of", x ," ", occurance_of_word)
    

    DICTIONARY = dict() 
    

    tok_list = vectorizer.get_feature_names_out(tr_text_list)
    for tok in tok_list:
        if tok in DICTIONARY:
            DICTIONARY[tok] += 1
        else:
            DICTIONARY[tok] = 1
    
    
    sorted_tokens = list(sorted(DICTIONARY, key=DICTIONARY.get, reverse=True))
    for w in sorted_tokens[:10]:
        print("%5d %s" % (DICTIONARY[w], w))
    # print(word_list)

    # print(coords_occur_list.toarray())
    print("end of operation")
    


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