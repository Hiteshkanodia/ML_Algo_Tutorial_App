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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap
import os
import time
def model(X_c1):
    print("hitesh")
    cmap_1 = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    y_c1=np.ones((X_c1.shape[0],1))
    # Splitting the dataset into the Training set and Test set


    X_c1_train, X_c1_test, y_c1_train, y_c1_test = train_test_split(X_c1, y_c1, test_size = 0.25, random_state = 0)
    
    plt.figure()
    plt.scatter(X_c1_train[:,0],X_c1_train[:,1],c=y_c1_train.ravel(),cmap='rainbow')
    plt.title("Data points of 1 classes")
    plt.xlabel('x1')
    plt.ylabel('x2')
    new_graph_name1 = "graph1" + str(time.time()) + ".png"
    for filename in os.listdir('static/'):
        if filename.startswith('graph'):  # not to remove other images
            os.remove('static/' + filename)
    plt.savefig('static/'+ new_graph_name1,bbox_inches='tight')


    model = LinearRegression().fit(X_c1_train, y_c1_train.ravel())
    # Predicting the Test set results
    y_pred = model.predict(X_c1_test)
    print("hello")
    print(y_pred)
    print("hello")

    accuracy = r2_score(y_c1_test, y_pred)
    print ("The Accuracy=",accuracy*100)
    
    x_min, x_max = X_c1_train[:, 0].min() - .5, X_c1_train[:, 0].max() + .5
    y_min, y_max = X_c1_train[:, 1].min() - .5, X_c1_train[:, 1].max() + .5
    hticks = np.linspace(x_min,x_max,.1)
    vticks  = np.linspace(y_min,y_max,.1)

    xx, yy = np.meshgrid(hticks,vticks)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.contourf(xx,yy,Z,cmap='bwr',alpha=0.2)
    
    # Visualising the Training set results
    #plt.scatter(X_c1_train[:, 0], X_c1_train[:, 1],y_c1_train,cmap=plt.cm.Paired)
    #plt.plot(X_c1_test[:,0],X_c1_test[:,1],y_pred,color='blue',linewidth=3)
    plt.title('Decision region using Linear Regression' )
    plt.xlabel('X1')
    plt.ylabel('X2')
    new_graph_name2 = "graph2" + str(time.time()) + ".png"
    plt.savefig('static/'+ new_graph_name2,bbox_inches='tight')
    plt.close()
    return (new_graph_name1,new_graph_name2,accuracy)