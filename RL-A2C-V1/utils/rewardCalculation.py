

class rewardCalculation:
    
    def __init__(self, positions, step, selectedAction, data, positionSwitch = False, stepSwith = True):
        
        self. positions = positions
        self. selectedAction = selectedAction
        self. step = step
        self. positionSwitch= positionSwitch
        self. stepSwith = stepSwith
        self. data = data
        
        
    
    def reward(self):
        
        if self. positionSwitch == True:
           if  self. positions["trade " + len(self. positions)]["closePrice"] != 0:
                return(self. positions["trade " + len(self. positions)]["profit"])
        
           else:
            
                return(0)
            
            
        
        if self. stepSwith == True:
            if self. selectedAction == "Sell" and self. data["Last"][self. step] > self. data["Last"][self. step + 1 ]:
                return(+1)
            
            if self. selectedAction == "Buy" and self. data["Last"][self. step] < self. data["Last"][self. step + 1 ]:
                return(+1)

            else:
                return(-1)
            
        
        
            
        