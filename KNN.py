#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Hitesh Kanodia,Jaideep Bhargava
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import os
import time
def model(k,X_c1,X_c2):
    cmap_1 = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    y_c1=np.ones((X_c1.shape[0],1))
    y_c2=np.zeros((X_c2.shape[0],1))
    # Splitting the dataset into the Training set and Test set


    X_c1_train, X_c1_test, y_c1_train, y_c1_test = train_test_split(X_c1, y_c1, test_size = 0.25, random_state = 0)
    X_c2_train, X_c2_test, y_c2_train, y_c2_test = train_test_split(X_c2, y_c2, test_size = 0.25, random_state = 0)
    X_train=np.concatenate((X_c1_train, X_c2_train), axis=0)
    y_train=np.concatenate((y_c1_train, y_c2_train), axis=0)

    X_test=np.concatenate((X_c1_test, X_c2_test), axis=0)
    y_test=np.concatenate((y_c1_test, y_c2_test), axis=0)

    plt.figure()
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train.ravel(),cmap='rainbow')
    plt.title("Data points of 2 classes")
    plt.xlabel('x1')
    plt.ylabel('x2')
    new_graph_name1 = "graph1" + str(time.time()) + ".png"
    for filename in os.listdir('static/'):
        if filename.startswith('graph'):  # not to remove other images
            os.remove('static/' + filename)
    plt.savefig('static/'+ new_graph_name1,bbox_inches='tight')

    K=int(k)
    print(K)
    print(str(K))
    # Fitting K-NN to the Training set

    model = KNeighborsClassifier(n_neighbors = K, metric = 'minkowski', p = 2)
    #The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
    model.fit(X_train, y_train.ravel())

    # Predicting the Test set results
    y_pred = model.predict(X_test)
    print(y_pred)
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Confusion Matrix=\n",cm)
    print ("The Accuracy=",accuracy*100)

    # Visualising the Training set results
    X_set, y_set = X_train, y_train
    plt.figure(1)
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))

    Z=model.predict(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)
    plt.contour(X1, X2, Z, cmap=cmap_1)
    plt.pcolormesh(X1, X2, Z,cmap=cmap_1)
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train.ravel(),cmap='rainbow') 
    print(str(K))
    plt.title('Decision region using KNN with K='+str(K) )
    plt.xlabel('X1')
    plt.ylabel('X2')
    new_graph_name2 = "graph2" + str(time.time()) + ".png"
    plt.savefig('static/'+ new_graph_name2,bbox_inches='tight')
    plt.close()
    return (new_graph_name1,new_graph_name2,accuracy,cm)