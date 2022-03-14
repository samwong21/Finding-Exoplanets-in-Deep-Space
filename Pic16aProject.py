#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sklearn as sklearn

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
    
 
        
class star_data:
    '''
    This class takes in train data in the form of a csv file, turns it into a pandas dataframe and uses it to train 
    decision tree and KNN models. Also includes methods to visualize decision trees and scores of both models. 
    
    Args:
        datafile: csv data file
        d: depth of decision tree
        n: number of neighbors for KNN
        
    Returns:
        None
        
    '''
   
    
    def __init__(self, datafile, d = 2, n = 2):
        self.df = pd.read_csv(datafile)
        self.T = tree.DecisionTreeClassifier(max_depth=d)
        self.knn = KNeighborsClassifier(n_neighbors=n)
    
    def split(self, test_size = 0.3):
        '''
        Splits data into predictor and outcome variables, then splits again into train and faux test data
        
        Args:
            test_size: percentage of data that will become faux test data
        Returns:
            self.X_train: predictor variables for train data set split
            self.X_test: predictor variables for test data set split
            self.y_train: outcome variables for train data set split
            self.y_test: outcome variables for test data set split
        
        '''
        X = self.df.drop(['LABEL'], axis=1)
        y = self.df['LABEL']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size)
        return self.X_train, self.X_test, self.y_train, self.y_test
     
    def fit_tree(self): 
        '''
        Args:
            None
        Returns:
            self.model_tree: model of decision tree based on newly split data 
        '''
        self.split() #plots decision tree model after calling fit_tree method to make model
        self.model_tree = self.T.fit(self.X_train, self.y_train)
        return self.model_tree
    

    def plot_tree(self):
        '''
        Plots visualization of decision tree.
        Args:
            None
        Returns:
            plt.show() : plots decision tree
        '''
        self.fit_tree()
        fig, ax = plt.subplots(1, figsize = (10, 10))
        p = tree.plot_tree(self.T, filled=True, feature_names = self.X_train.columns)
        plt.show()
        
        
    def fit_knn(self):
        '''
        Fits KNN model to X_train and y_train.
        Args:
            None
        Returns:
            self.model_knn: KNN model fitted to train data
        '''
        self.split()
        self.model_knn = self.knn.fit(self.X_train, self.y_train)
        return self.model_knn
              
        
    def tree_train_score(self):
        '''
        Computes tree model accuracy of predicting y_train from X_train.
        Args:
            None
        Returns:
            self.T.score(self.X_train, self.y_train): score of tree model on train data
        
        '''
        return self.T.score(self.X_train, self.y_train)
   
    def tree_test_score(self):
        '''
        Computes tree model accuracy of predicting y_test from X_test.
        Args:
            None
        Returns:
            self.T.score(self.X_train, self.y_train): score of tree model on fake test data
        
        '''
        return self.T.score(self.X_test, self.y_test)
    
    
    def all_tree_scores(self):
        '''
        Checks if model is a decision tree than prints both test score and train score
        Args:
            None
        Returns (prints):
            self.T.score(self.X_train, self.y_train): score of tree model on train data
            self.T.score(self.X_train, self.y_train): score of tree model on fake test data
        
        '''
        #first instance of exception handling, ensures that the model is a decision tree
        if type(self.model_tree) == sklearn.tree._classes.DecisionTreeClassifier:
                print("Train score for tree model:", self.tree_train_score() )
                print("Fake Test score for tree model:", self.tree_test_score() )
        else: 
            raise TypeError ("not valid decision tree model type, try another model")
  
    def knn_train_score(self):
        '''
        Computes KNN model accuracy of predicting y_train from X_train.
        Args:
            None
        Returns:
            self.knn.score(self.X_train, self.y_train): score of tree model on train data
        
        '''
        return self.knn.score(self.X_train, self.y_train)
    
    def knn_test_score(self):
        '''
        Computes KNN model accuracy of predicting y_test from X_test.
        Args:
            None
        Returns:
            self.knn.score(self.X_train, self.y_train): score of KNN model on fake test data
        
        '''
        return self.knn.score(self.X_test, self.y_test)
    
    def all_knn_scores(self):
        '''
        Checks if model is a KNN than prints both test score and train score
        Args:
            None
        Returns (prints):
            self.knn.score(self.X_train, self.y_train): score of KNN model on train data
            self.knn.score(self.X_train, self.y_train): score of KNN model on fake test data
        
        '''
        #second instance of exception handling, ensures that the model is a KNN
        if type(self.model_knn) == sklearn.neighbors._classification.KNeighborsClassifier:
                print("Train score for KNN model:", self.knn.score(self.X_train, self.y_train))
                print("Test score for KNN model:", self.knn.score(self.X_test, self.y_test))
        else: 
            raise TypeError ("not valid KNN model type, try another model")
        

