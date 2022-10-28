# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 20:17:08 2022

@author: a.h
"""

import numpy as np
import random

class updatinQtable:
    
    def __init__(self, memory, batchSize, Q, gamma, nextState):
        self. batchSize = batchSize
        self. memory = memory
        self. Q = Q
        self. gamma = gamma
        self. nextState = nextState
        
    
    
    def updating(self):
        
       
        
        batch = []
        
        if len(self. memory) == self. batchSize:
            
              batch = self. memory 
        
        else:
            
          randomSelected = random.sample(range(0, len(self. memory)), self. batchSize)
          
          for _ in  randomSelected:
              batch.append(self.memory[_])
              
    
        self. nextState = np.reshape(self. nextState,(1, np.shape(self. nextState)[0]))
        
        for state, action, reward, next_state in batch:
        
          reward = reward + self.gamma * np.amax(self. Q.predict(self. nextState)[0])
          
          
          state = np.reshape(state,(1, np.shape(state)[0]))
          target = self. Q. predict(state)
          target[0][action] = reward
      
          self. Q. fit(state, target, epochs=1, verbose=0)

        return(self. Q)
    
    
    
    
    
    