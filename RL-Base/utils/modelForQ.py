# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:44:36 2022

@author: a.h
"""

import numpy as np

class modelForQ:
    
    def __init__(self, xTrain, yTrain):
        self. xTrain = xTrain
        self. yTrain = yTrain
    
    
    def fullyConnectedModel(self):
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten
        
        model = Sequential()
        # model.add(Flatten(input_shape = (1, np.shape(self. xTrain)[1])))
        model.add(Dense(24, input_shape = (np.shape(self. xTrain)[1],) , activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(np.shape(self. yTrain)[0], activation='linear'))
        # compile the keras model
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return(model)
    
    
    
    
    
                
        
        