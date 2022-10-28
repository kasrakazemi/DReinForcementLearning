################################# import libs ##############################
import random
import numpy as np


class policy:
    
    def __init__(self, Actor, actions, state, epsilon):

        self. Actor = Actor
        self. actions = actions
        self. state = state
        self. epsilon = epsilon
        
    
    def actionSelection(self):
        
        # if random.uniform(0, 1) < self. epsilon:
            
        #     selectedAction = self. actions[random.randint(0,len(self. actions)-1)]
            
        #     return selectedAction
       
        
        # else:

        self. current_state = np.reshape(self. state,(1, np.shape(self.state)[0]))
        
        actionsPowerNumpy = self. Actor.predict (self. current_state,verbose=0)
        actionsPower = list()
        for _ in actionsPowerNumpy:
            actionsPower.append(_)
            
        selectedAction = self. actions[actionsPower.index(max(actionsPower))]
        
        
        return selectedAction
        
        

        
        
        