from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use('seaborn-v0_8') # pretty matplotlib plots
# sns.set('notebook', style='whitegrid', font_scale=1.25)


# from sklearn.model_selection import KFold
# from sklearn.metrics import root_mean_squared_error
# from sklearn import preprocessing
# import sklearn as sk
# import textwrap

# import sklearn.linear_model
# import sklearn.metrics
# from sklearn.model_selection import train_test_split, KFold
# from scipy.stats import trimboth


def load_and_process_data(dir_path, x_tr_path, y_tr_path):
    '''
    Args
    ----
    dir_path : string
        The directory path to the files to open
    x_tr_path : string
        The file path to the inputs of the model
    y_tr_path : string
        The file path to the ground-truth labels

    Returns
    -------
    x_train_df : list
        List of all samples' 'text' col opened from file at x_tr_path
    y_train_df : list
        List of all samples' 'Course Label' col opened from file at y_tr_path
    '''

    x_train_df = pd.read_csv(os.path.join(dir_path, x_tr_path))
    y_train_df = pd.read_csv(os.path.join(dir_path, y_tr_path))

    N1, n1_cols = x_train_df.shape
    N2, n2_cols = y_train_df.shape

    assert(N1 == N2)

    x_tr_raw = x_train_df['text'].tolist()
    y_tr_raw = y_train_df['Coarse Label'].tolist()


    return x_tr_raw, y_tr_raw

def plot_c_vs_ROC(grid_search, param_grid):
    C_values = param_grid['clf__C']
    train_scores = grid_search.cv_results_['mean_train_score']
    test_scores = grid_search.cv_results_['mean_test_score']

    # Plotting the performance
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_scores, label="Train AUC", marker="o")
    plt.plot(C_values, test_scores, label="Test AUC", marker="o")

    plt.xscale("log")  # Log scale for better visualization
    plt.xlabel("C")
    plt.ylabel("AUC Score")
    plt.title("C value vs AUC Score")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x_tr_raw, y_tr_raw = load_and_process_data('data_readinglevel', 'x_train.csv', 'y_train.csv')
    y_tr_N = (np.array(y_tr_raw) == 'Key Stage 4-5').astype(int)

    # Creating pipeline that vectorizes x_tr_raw and preforms log regression
    pipeline = Pipeline([
        # ('vect', CountVectorizer(lowercase=True, min_df=1)), # TODO Experiement with preprocessing here
        ('vect', CountVectorizer(lowercase=True, min_df = 1)),
        ('clf', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000))
    ])



    param_grid = {
        # 'clf__C' : [0.625055192527395],
        'clf__C': np.logspace(-5, 5, 50)

        # 'clf__C': np.logspace(-10, 10, 50) # TODO Find better values here 
        # 'clf__C': np.logspace(-5.857143, 4.142857,11)

        ##alpha_list = np.logspace(-10, 6, 17)
    }

    # Picking best hyperparameters with CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # This preforms betters than regular KFolds
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', return_train_score=True)

    grid_search.fit(x_tr_raw, y_tr_N)

    print("this is the score with kFolds")
    print("Best hyperparameter C:", grid_search.best_params_)
    print("Best CV AUROC:", grid_search.best_score_)


    # Training best model and predicting on test set
    best_model = grid_search.best_estimator_
    best_model.fit(x_tr_raw, y_tr_N)

    x_te_raw = pd.read_csv(os.path.join('data_readinglevel', "x_test.csv"))['text'].tolist()
    # print(x_te_raw)


    yproba1_te = best_model.predict_proba(x_te_raw)[:, 1]
    yproba1_tr = best_model.predict_proba(x_tr_raw)[:, 1]

    # print("tr", yproba1_tr)
    # print("te", yproba1_te)


    # Graphing Stuff
    results = grid_search.cv_results_

  

    # print(results)

    plot_c_vs_ROC(grid_search, param_grid)





    # Saving x_test positive probabilities to a file
    np.savetxt("yproba1_test.txt", yproba1_te, fmt="%.6f")
    np.savetxt("yproba1_train.txt", yproba1_tr, fmt="%.6f")


    



'''
    CountVectorizer(lowercase=True, min_df=10)
    'clf__C': [0.01, 0.1, 1, 10, 100]
    0.7859991537570853

    CountVectorizer(lowercase=True, min_df=10)
    np.logspace(-10, 6, 20)
    0.7861957353758151

    CountVectorizer(lowercase=True, min_df=10)
    clf__C': np.logspace(-10, 6, 50)
    0.7858031039027337

    CountVectorizer(lowercase=True)
    clf__C : 0.1389495494373136
    0.8135651630667169

    ('vect', CountVectorizer(lowercase=True)
     'clf__C': np.logspace(-5.857143, 4.142857,11)
    0.8182692670411618
    
    max_df = 0.5 
    Best CV AUROC: 0.8169414974473194

    ('vect', CountVectorizer(lowercase=True, max_df=0.7))
    0.8173088690175188

    ('vect', CountVectorizer(lowercase=True, min_df=5))
    0.8032191243189871

    this is the score with kFolds
Best hyperparameter C: {'clf__C': 10.0} tdif vectorizor
Best CV AUROC: 0.8348367960217086

%                                       

'''


    


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