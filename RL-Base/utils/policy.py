# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:14:30 2022

@author: a.h
"""

import random
import numpy as np



class policy:
    
    def __init__(self, Q, actions, state, epsilon):
        self. Q = Q
        self. actions = actions
        self. state = state
        self. epsilon = epsilon
        
    
    def actionSelection(self):
        
        if random.uniform(0, 1) < self. epsilon:
            
            selectedAction = self. actions[random.randint(0,len(self. actions)-1)]
            
            return selectedAction
       
        
        else:
            
            self. state = np.reshape(self. state,(1, np.shape(self. state)[0]))
            
            actionsPowerNumpy = self. Q.predict (self. state)
            actionsPower = list()
            for _ in actionsPowerNumpy:
                actionsPower.append(_)
            selectedAction = self. actions[actionsPower.index(max(actionsPower))]
          
          
            return selectedAction
        
        
        
        
        