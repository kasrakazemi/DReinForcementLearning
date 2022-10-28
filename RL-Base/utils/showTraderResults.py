# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:36:27 2022

@author: a.h
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:30:05 2022

@author: Termite021
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:57:02 2021

@author: a.h
"""

import matplotlib.pyplot as plt
import numpy as np

class showTraderResults:
    
    def __init__(self, trades, realTestOutput):
        self. trades = trades
        self. realTestOutput = realTestOutput
      
    
    def statementResults(self):
        trades_ = self. trades
       
        numberOfBuy = 0
        numberOfSell = 0
        numberOfPositveSell = 0
        numberOfPositiveBuy = 0
        profitInBuyTrades = 0
        profitInSellTrades = 0
        netProfit = 0
        totallProfit = 0
        totallLoss = 0
        numberOfLossBuy = 0
        lossInBuyTrades = 0
        numberOfLossSell = 0
        lossInSellTrades = 0
        netBuy = 0
        netSell = 0
        listOfMargin = list()
        
        for i in range(1, len(trades_)):
            
             ####################################### positive profit section ####################
            listOfMargin.append(trades_["trade " + str(i)]["margin"])
            
            
            if trades_["trade " + str(i)]["profit"] > 0:
                totallProfit += trades_["trade " + str(i)]["profit"]
            
            
            
            
            ####################################### loss section #################################
            if trades_["trade " + str(i)]["profit"] < 0:
                totallLoss += trades_["trade " + str(i)]["profit"]
            
            
            
            
            
            
            ################################ buy section ############################################
            if trades_["trade " + str(i)]["type"] == "buy":
               numberOfBuy += 1
               if trades_["tarde " + str(i)]["profit"] >0:
                   numberOfPositiveBuy += 1
                   profitInBuyTrades  += trades_["trade " + str(i)]["profit"]
                
                
               if trades_["trade " + str(i)]["profit"] < 0:
                   numberOfLossBuy += 1
                   lossInBuyTrades  += trades_["trade " + str(i)]["profit"]
                   

                   
                   
                   
                   
        ############################################### sell section #####################################
            if trades_["trade " + str(i)]["type"] == "sell":
                numberOfSell += 1

                if trades_["trade " + str(i)]["profit"] >0:
                    numberOfPositveSell += 1
                    profitInSellTrades += trades_["trade " + str(i)]["profit"]
                
                
                if trades_["trade " + str(i)]["profit"] < 0 :
                    numberOfLossSell += 1
                    lossInSellTrades += trades_["trade " + str(i)]["profit"]
             
                
             
                
             
                
     ################################## total section ###################################################3
        netProfit += trades_["trade " + str(i)]["profit"]
        netBuy = profitInBuyTrades - lossInBuyTrades
        netSell = profitInSellTrades - lossInSellTrades
        
        winRate = ((numberOfPositiveBuy + numberOfPositveSell) / len(trades_))*100
        lossRate = ((numberOfLossSell + numberOfLossBuy) / len(trades_))*100
        
        finalResult = dict()
        
        finalResult["numberOfTrades"] = len(trades_)
        finalResult["numberOfBuy"] = numberOfBuy
        finalResult["numberOfSell"] = numberOfSell
        finalResult["numberOfPositveSell"] = numberOfPositveSell
        finalResult["numberOfPositiveBuy"] = numberOfPositiveBuy
        finalResult["profitInBuyTrades"] = profitInBuyTrades
        finalResult["profitInSellTrades"] = profitInSellTrades
        finalResult["netProfit"] = netProfit
        finalResult["totallProfit"] = totallProfit
        finalResult["totallLoss"] = totallLoss
        finalResult["numberOfLossBuy"] = numberOfLossBuy
        finalResult["lossInBuyTrades"] = lossInBuyTrades
        finalResult["numberOfLossSell"] = numberOfLossSell
        finalResult["lossInSellTrades"] = lossInSellTrades
        finalResult["netBuy"] = netBuy
        finalResult["netSell"] = netSell
        finalResult["winRate"] = winRate
        finalResult["lossRate"] = lossRate
        
        
        return(listOfMargin, finalResult)
    

    def  showAsFigure(self):
        
        listOfMargin,_ = self. statementResults() 
        plt.figure(4)
        maxMargin = max(listOfMargin)
        minMargin = min(listOfMargin)
        plt.title("max margin is:   " + str(maxMargin) + "min margin is: " + str(minMargin))
        t = np.linspace(0,len(listOfMargin)-1, len(listOfMargin))
        plt.plot(t, listOfMargin, '-ok', color="blue", linewidth = 3)
        
        trades_ = self. trades
        _, finalResult = self. statementResults()
        plt.figure(10)
        plt.title("total net profit is:  " + str(finalResult["netProfit"]))
        
        # fplt.plot(df_y, figratio=(20,12), type = 'candle', title = 'price_real'+a, mav = (3), style= 'yahoo')

        
        t = np.linspace(0,len(self. realTestOutput)-1, len(self. realTestOutput))
        plt.plot(t, self. realTestOutput, color="blue"
                 , linewidth = 3)
        
        for i in range(1, len(trades_)):
            openPrice = trades_["trade "+str(i)]["openPrice"]
            tOpen = t[int(trades_["trade "+str(i)]["openStep"])] 
            
            if trades_["trade "+str(i)]["type"] == "buy":
                color = "green"
            else:
                color = "red"
            
            if not openPrice == 0:
                 plt.scatter(tOpen, openPrice, c = color, s = 100)

            closePrice = trades_["trade "+str(i)]["closePrice"]
            tClose = t[int(trades_["trade "+str(i)]["closeState"])] 
            if not closePrice == 0:
                plt.scatter(tClose, closePrice, c = color, s =100)
            
            if not closePrice == 0:
                plt.plot([tOpen, tClose],[openPrice, closePrice], color="yellow"
                 , linewidth = 6)
   
    
    
    def final(self):
        
        _, statements = self. statementResults()
        
        self. showAsFigure()
        
        return(statements)
        
        
        
        
            
        
        
                
                
        
                
            
            
            
            
        
        
    
                    
            
    
    