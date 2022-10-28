

import random


class openOrClosePosition:
    
    def __init__(self, positions, selectedAction, step, data, margine, Sl, Tp, levrage):
        self. margin = margine
        self. positions = positions
        self. selectedAction = selectedAction
        self. step = step
        self. data = data
        self. Sl = Sl
        self. Tp = Tp
        self. levrage = levrage
        
    
    def openingAposition(self, margin):
        actions = ["Buy" , "Sell"]
        if self. selectedAction == "Hold":
            selectedAction = actions[random.randint(0,len(actions)-1)]
        
        else:
            selectedAction = self. selectedAction
        
            
        status = dict()
        status["openPrice"] = self. data["close"][self. step]
        status["type"] = selectedAction
        status["closePrice"] = 0
        status["profit"] = 0
        status["openStep"] = self. step
        status["closeState"] = 0
        status["margin"] = margin
        status["firstMargin"] =margin
        status["lastClose"] = self. data["close"][self. step]
        self. positions["trade " + str(len(self. positions)+1)] = status
        
        return(self. positions)
        
        
    
    def closingAposition(self):
        
        
        if self. positions["trade " + str(len(self. positions))]["type"] == "Buy":

            stopLoss = self. positions["trade " + str(len(self. positions))]["openPrice"] - (self. positions["trade " + str(len(self. positions))]["openPrice"] * self. Sl)
            takeProfit = self. positions["trade " + str(len(self. positions))]["openPrice"] + (self. positions["trade " + str(len(self. positions))]["openPrice"] * self. Tp)
            
            if self. selectedAction == "Sell":
                self. positions["trade " + str(len(self. positions))]["closePrice"] =  self. data["close"][self. step]
                self. positions["trade " + str(len(self. positions))]["profit"] =  self. data["close"][self. step]  -   self. positions["trade " + str(len(self. positions))]["openPrice"]       
                self. positions["trade " + str(len(self. positions))]["closeState"] = self. step
                self. positions["trade " + str(len(self. positions))]["margin"] += self. positions["trade " + str(len(self. positions))]["profit"]   
                
            
            if  self. data["low"][self. step] <= stopLoss:
                self. positions["trade " + str(len(self. positions))]["closePrice"] =  self. data["close"][self. step]
                self. positions["trade " + str(len(self. positions))]["profit"] =  self. data["close"][self. step]  -   self. positions["trade " + str(len(self. positions))]["openPrice"]       
                self. positions["trade " + str(len(self. positions))]["closeState"] = self. step
                self. positions["trade " + str(len(self. positions))]["margin"] += self. positions["trade " + str(len(self. positions))]["profit"]   
            
            if self. data["high"][self. step] >= takeProfit:
                self. positions["trade " + str(len(self. positions))]["closePrice"] =  self. data["close"][self. step]
                self. positions["trade " + str(len(self. positions))]["profit"] =  self. data["close"][self. step]  -   self. positions["trade " + str(len(self. positions))]["openPrice"]       
                self. positions["trade " + str(len(self. positions))]["closeState"] = self. step
                self. positions["trade " + str(len(self. positions))]["margin"] += self. positions["trade " + str(len(self. positions))]["profit"]   
            
        
        
        if self. positions["trade " + str(len(self. positions))]["type"] == "Sell":
            
            stopLoss = self. positions["trade " + str(len(self. positions))]["openPrice"] + (self. positions["trade " + str(len(self. positions))]["openPrice"] * self. Sl)
            takeProfit = self. positions["trade " + str(len(self. positions))]["openPrice"] - (self. positions["trade " + str(len(self. positions))]["openPrice"] * self. Tp)
            
            
            
            if self. selectedAction == "Buy":
                self. positions["trade " + str(len(self. positions))]["closePrice"] =  self. data["close"][self. step]
                self. positions["trade " + str(len(self. positions))]["profit"] =  self. positions["trade " + str(len(self. positions))]["openPrice"] - self. data["close"][self. step]        
                self. positions["trade " + str(len(self. positions))]["closeState"] = self. step
                self. positions["trade " + str(len(self. positions))]["margin"] += self. positions["trade " + str(len(self. positions))]["profit"]   
            
            if self. data["high"][self. step] >= stopLoss:
                self. positions["trade " + str(len(self. positions))]["closePrice"] =  self. data["close"][self. step]
                self. positions["trade " + str(len(self. positions))]["profit"] =  self. positions["trade " + str(len(self. positions))]["openPrice"] - self. data["close"][self. step]        
                self. positions["trade " + str(len(self. positions))]["closeState"] = self. step
                self. positions["trade " + str(len(self. positions))]["margin"] += self. positions["trade " + str(len(self. positions))]["profit"]   
            
            
            
            if self. data["low"][self. step] <= takeProfit:
                self. positions["trade " + str(len(self. positions))]["closePrice"] =  self. data["close"][self. step]
                self. positions["trade " + str(len(self. positions))]["profit"] =  self. positions["trade " + str(len(self. positions))]["openPrice"] - self. data["close"][self. step]        
                self. positions["trade " + str(len(self. positions))]["closeState"] = self. step
                self. positions["trade " + str(len(self. positions))]["margin"] += self. positions["trade " + str(len(self. positions))]["profit"]   
            
            
            
        return(self. positions)
    
    
    
    def tradeCheck(self):
        
        if len(self. positions) == 0 or self. positions["trade " + str(len(self. positions))]["closePrice"] != 0:
            if len(self. positions) == 0:
                margin = self. margin
            else:
                margin = self. positions["trade " + str(len(self. positions))]["margin"]
            positions = self. openingAposition(margin)
            
            return(positions)
        
        
        if len(self. positions) != 0 and self. positions["trade " + str(len(self. positions))]["closePrice"] == 0:
            positions = self. closingAposition()
            
            return(positions)
        
        else:
            if self. positions["trade " + str(len(self. positions))]["type"] == self. selectedAction or self. selectedAction == "hold":
                pass
            
            else:
                return(self. closingAposition())
    
    

            
        
        
    
    
        
        
        
        
        
        
        