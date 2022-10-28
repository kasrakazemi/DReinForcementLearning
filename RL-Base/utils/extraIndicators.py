# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:20:23 2022

@author: a.h
"""
import pandas_ta as ta
import numpy as np


class extraIndicators:
    
    def __init__(self,data):
        

        self. data = data
     
        
    def supertrend(self, df):
        
        df["sti0"] = ta.supertrend(df['High'], df['Low'], df['Close']).values[:,0]
        df["sti1"] = ta.supertrend(df['High'], df['Low'], df['Close']).values[:,1]
        df["sti2"] = ta.supertrend(df['High'], df['Low'], df['Close']).values[:,2]
        df["sti3"] = ta.supertrend(df['High'], df['Low'], df['Close']).values[:,3]

        return df

    def RVIindicator(self, df):

        df["RVIindicator"] = ta.rvi(df["close"], high = df["high"], low = df["low"])

        return df

    def SMIindicator(self, df):

        df["SMI0"] = ta.smi(df["close"]).values[:,0]
        df["SMI1"] = ta.smi(df["close"]).values[:,1]
        df["SMI2"] = ta.smi(df["close"]).values[:,2]
        
        return df

    def RVGIindicator(self, df):

        df["RVGI0"] = ta.rvgi(df["open"], df["high"], df['low'], df['close']).values[:,0]
        df["RVGI1"] = ta.rvgi(df["open"], df["high"], df['low'], df['close']).values[:,1]

        return df



    def STDVindicator(self, df):

        df["STDV"] = ta.stdev(df["close"])

        return(df)

    def HVindicator(self, df):

        tradingDays = 252
        returns = np.log(df['close']/df['close'].shift(1))
        df["HV"] = returns.rolling(window = tradingDays).std()*np.sqrt(tradingDays)

        return df
    
    
    def MomentumIndicator(self, df):
        
        length_KC = 20
        length = 20

        highest = df['high'].rolling(window = length_KC).max()
        lowest = df['low'].rolling(window = length_KC).min()
        m1 = (highest + lowest) / 2
        m_avg = df['close'].rolling(window=length).mean()
        df['Momentum'] = (df['close'] - (m1 + m_avg)/2)
        fit_y = np.array(range(0,length_KC))
        df['Momentum'] = df['Momentum'].rolling(window = length_KC).apply(lambda x : np.polyfit(fit_y, x, 1)[0] * (length_KC-1) +np.polyfit(fit_y, x, 1)[1], raw=True)
        
        return df
    
    
    def final(self):
        df = self. data
        
        df = self. RVIindicator(df)
        df = self. SMIindicator(df)
        df = self. RVGIindicator(df)
        
        df = self. STDVindicator(df)
        df = self. HVindicator(df)
        
        df = self. MomentumIndicator(df)
        
        return(df)
        
        
        
        
        
        
    



        


    















        
        

