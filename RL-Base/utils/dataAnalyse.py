
################### imports ######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from keras.utils import np_utils
import pandas as pd
from ta import *
from ta.utils import dropna
from ta.trend import MassIndex
import pickle as pk
from sqlalchemy import create_engine
import datetime
from .extraIndicators import extraIndicators

class dataAnalyse:
    
    def __init__(self, data, inputDays, testDays):

        self. df = data
        self. inputDays = inputDays
        self. testDays = testDays

    
    def indicators(self):
        
        df2 = self. df
        df2.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN value
        df2 = add_volatility_ta(
            df2, high="high", low="low", close="close")
        df2 = add_trend_ta(df2,
            high="high", low="low", close="close")
        
        
        df2 = add_momentum_ta(df2,
            high="high", low="low", close="close", volume = "volume")
        
        extraIndicators_ = extraIndicators(df2)
        
        df2 = extraIndicators_.final()
        
        
        df2.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN value
        df2.reset_index(inplace=True)
        df2.drop(['index'], axis=1, inplace=True)
        

        
        df2['open'] = (df2['open'].pct_change())*20# Create arithmetic returns column
        df2['high'] = (df2['high'].pct_change())*20# Create arithmetic returns column
        df2['low'] = (df2['low'].pct_change())*20 # Create arithmetic returns column
        df2['close'] = (df2['close'].pct_change())*20 #Create arithmetic returns column
        
        
        df2["BB_Base"] = (df2['BB_Base'].pct_change())*100
        df2["BB_H"] = (df2['BB_H'].pct_change())*100
        df2["BB_L"] = (df2['BB_L'].pct_change())*100
        df2["KeltnerChannel"] = (df2['KeltnerChannel'].pct_change())*100
        df2["KeltnerChannel_high"] = (df2['KeltnerChannel_high'].pct_change())*100
        df2["KeltnerChannel_low"] = (df2['KeltnerChannel_low'].pct_change())*100
        
        
        df2["DONCHAIN_LOW"] = (df2['DONCHAIN_LOW'].pct_change())*10
        df2["DONCHAIN_HIGH"] = (df2['DONCHAIN_HIGH'].pct_change())*10
        df2["DONCHAIN_BASE"] = (df2['DONCHAIN_BASE'].pct_change())*10
        df2["ATR"] = (df2['ATR'].pct_change())*5
        
        
        df2["sma250"] = (df2['sma250'].pct_change())*100
        
        df2["WMA126"] = (df2['WMA126'].pct_change())*100
        df2["WMA63"] = (df2['WMA63'].pct_change())*100
        
        df2["ema10"] = (df2['ema10'].pct_change())*100
        df2["ema21"] = (df2['ema21'].pct_change())*100
        df2["ema50"] = (df2['ema50'].pct_change())*100
        df2["ema55"] = (df2['ema55'].pct_change())*100
        
        
        
        
        df2["PSAR"] = (df2['PSAR'].pct_change())*10
        df2["trend_ichimoku_conv"] = (df2['trend_ichimoku_conv'].pct_change())*50
        df2["trend_ichimoku_base"] = (df2['trend_ichimoku_base'].pct_change())*50
        df2["trend_ichimoku_a"] = (df2['trend_ichimoku_a'].pct_change())*50
        df2["trend_ichimoku_b"] = (df2['trend_ichimoku_b'].pct_change())*50
        
        
        df2.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values)
        df2.reset_index(inplace=True)
        df2.drop(['index'], axis=1, inplace=True)
        
        
        return(df2)
  
    def dataCleaning(self):
        df1 = self. indicators()
        df = pd.DataFrame(df1)
        df["time"] = 0
        # date = pd.to_datetime(df['time']).dt.date
        df.drop(['time'], axis=1, inplace=True)
        df["pair"] = 0 
        df.drop(['pair'], axis=1, inplace=True)       
        df['profit'] = df['close'] - df['open']
        df['label'] = 0
        for i in range(0,len(df)):
            if (df['profit'][i]>= 0):
                df['label'][i] = 1
       
        output = df["close"]
        return(df, output)



    def normalization(self):
        
        cleanedDf = self. dataCleaning()[0]
        normalizedDf = cleanedDf
        featurSetpoints = dict()
        
        ########################## max and min calculation ############################################
        
        featurSetpoints["MaxMacdMacd"] = max(normalizedDf["MACD_MACD"])
        featurSetpoints["MinMacdMacd"] = min(normalizedDf["MACD_MACD"])
        featurSetpoints["MaxMACD_Signal"] = max(normalizedDf["MACD_Signal"])
        featurSetpoints["MinMACD_Signal"] = min(normalizedDf["MACD_Signal"])
        featurSetpoints["MaxMACD_Histogeram"] = max(normalizedDf["MACD_Histogeram"])
        featurSetpoints["MinMACD_Histogeram"] = min(normalizedDf["MACD_Histogeram"])
 
        featurSetpoints["MaxADXandDI"] = max(normalizedDf["ADXandDI"])
        featurSetpoints["MinADXandDI"] = min(normalizedDf["ADXandDI"])
        
        featurSetpoints["MaxADXandDI_pos"] = max(normalizedDf["ADXandDI_pos"])
        featurSetpoints["MinADXandDI_pos"] = min(normalizedDf["ADXandDI_pos"])
        
        featurSetpoints["MaxADXandDI_neg"] = max(normalizedDf["ADXandDI_neg"])
        featurSetpoints["MinADXandDI_neg"] = min(normalizedDf["ADXandDI_neg"])
        
        featurSetpoints["MaxCCI"] = max(normalizedDf["CCI"])
        featurSetpoints["MinCCI"] = min(normalizedDf["CCI"])
        
        featurSetpoints["MaxROC"] = max(normalizedDf["ROC"])
        featurSetpoints["MinROC"] = min(normalizedDf["ROC"])
        
        featurSetpoints["MaxRVIindicator"] = max(normalizedDf["RVIindicator"])
        featurSetpoints["MinRVIindicator"] = min(normalizedDf["RVIindicator"])
        
        
        featurSetpoints["MinSMI0"] = min(normalizedDf["SMI0"])
        featurSetpoints["MinSMI1"] = min(normalizedDf["SMI1"])
        featurSetpoints["MinSMI2"] = min(normalizedDf["SMI2"])
        
        featurSetpoints["MaxSMI0"] = min(normalizedDf["SMI0"])
        featurSetpoints["MaxSMI1"] = min(normalizedDf["SMI1"])
        featurSetpoints["MaxSMI2"] = min(normalizedDf["SMI2"])
        
        featurSetpoints["MaxRVGI0"] = max(normalizedDf["RVGI0"])
        featurSetpoints["MinRVGI0"] = min(normalizedDf["RVGI0"])
        
        featurSetpoints["MaxRVGI1"] = max(normalizedDf["RVGI1"])
        featurSetpoints["MinRVGI1"] = min(normalizedDf["RVGI1"])
        
        featurSetpoints["MaxSTDV"] = max(normalizedDf["STDV"])
        featurSetpoints["MinSTDV"] = min(normalizedDf["STDV"])
        
        
        featurSetpoints["MaxHV"] = max(normalizedDf["HV"])
        featurSetpoints["MinHV"] = min(normalizedDf["HV"])
        
        
        featurSetpoints["MaxMomentum"] = max(normalizedDf["Momentum"])
        featurSetpoints["MinMomentum"] = min(normalizedDf["Momentum"])
        
        
        
        #################################################################### normalization ##############################################################
        

        normalizedDf["RSI"] = ((normalizedDf["RSI"] / 100) -0.5) * 2
        normalizedDf["STOCH"] = ((normalizedDf["STOCH"] / 100) -0.5) * 2
        normalizedDf["STOCH_signal"] = ((normalizedDf["STOCH_signal"] / 100) - 0.5) * 2
        normalizedDf["william"] = ((normalizedDf["william"] / -100) - 0.5 ) * 2
        
        
        normalizedDf["RVIindicator"] = ((normalizedDf["RVIindicator"] - min(normalizedDf["RVIindicator"])) / (max(normalizedDf["RVIindicator"]) - min(normalizedDf["RVIindicator"])) - 0.5 ) *2 
        
        normalizedDf["SMI0"] = ((normalizedDf["SMI0"] - min(normalizedDf["SMI0"])) / (max(normalizedDf["SMI0"]) - min(normalizedDf["SMI0"])) - 0.5 ) * 2
        normalizedDf["SMI1"] = (((normalizedDf["SMI1"] - min(normalizedDf["SMI1"])) / (max(normalizedDf["SMI1"]) - min(normalizedDf["SMI1"]))) - 0.5 ) * 2
        normalizedDf["SMI2"] = (((normalizedDf["SMI2"] - min(normalizedDf["SMI2"])) / (max(normalizedDf["SMI2"]) - min(normalizedDf["SMI2"]))) - 0.5 ) * 2
        
        normalizedDf["RVGI0"] = (((normalizedDf["RVGI0"] - min(normalizedDf["RVGI0"])) / (max(normalizedDf["RVGI0"]) - min(normalizedDf["RVGI0"]))) - 0.5 ) * 2
        normalizedDf["RVGI1"] = (((normalizedDf["RVGI1"] - min(normalizedDf["RVGI1"])) / (max(normalizedDf["RVGI1"]) - min(normalizedDf["RVGI1"]))) - 0.5 ) * 2
        
        normalizedDf["STDV"] = (((normalizedDf["STDV"] - min(normalizedDf["STDV"])) / (max(normalizedDf["STDV"]) - min(normalizedDf["STDV"]))) - 0.5 ) * 2
        
        normalizedDf["HV"] = (((normalizedDf["HV"] - min(normalizedDf["HV"])) / (max(normalizedDf["HV"]) - min(normalizedDf["HV"]))) - 0.5 ) * 2
        
        normalizedDf["Momentum"] = (((normalizedDf["Momentum"] - min(normalizedDf["Momentum"])) / (max(normalizedDf["Momentum"]) - min(normalizedDf["Momentum"]))) - 0.5 ) * 2
        
        
        
        
        normalizedDf["ROC"] = (((normalizedDf["ROC"] - min(normalizedDf["ROC"])) / (max(normalizedDf["ROC"]) - min(normalizedDf["ROC"]))) - 0.5 ) * 2
        
        normalizedDf["CCI"] = (((normalizedDf["CCI"] - min(normalizedDf["CCI"])) / (max(normalizedDf["CCI"]) - min(normalizedDf["CCI"]))) - 0.5 ) * 2
        normalizedDf["ADXandDI"] = (((normalizedDf["ADXandDI"] - min(normalizedDf["ADXandDI"])) / (max(normalizedDf["ADXandDI"]) - min(normalizedDf["ADXandDI"]))) - 0.5 ) * 2
        normalizedDf["ADXandDI_pos"] = (((normalizedDf["ADXandDI_pos"] - min(normalizedDf["ADXandDI_pos"])) / (max(normalizedDf["ADXandDI_pos"]) - min(normalizedDf["ADXandDI_pos"]))) - 0.5 ) * 2
        normalizedDf["ADXandDI_neg"] = (((normalizedDf["ADXandDI_neg"] - min(normalizedDf["ADXandDI_neg"])) / (max(normalizedDf["ADXandDI_neg"]) - min(normalizedDf["ADXandDI_neg"]))) - 0.5 ) * 2

        normalizedDf["MACD_Histogeram"] = (((normalizedDf["MACD_Histogeram"] - min(normalizedDf["MACD_Histogeram"])) / (max(normalizedDf["MACD_Histogeram"]) - min(normalizedDf["MACD_Histogeram"]))) - 0.5 ) * 2
        normalizedDf["MACD_Signal"] = (((normalizedDf["MACD_Signal"] - min(normalizedDf["MACD_Signal"])) / (max(normalizedDf["MACD_Signal"]) - min(normalizedDf["MACD_Signal"]))) - 0.5 ) * 2
        normalizedDf["MACD_MACD"] = (((normalizedDf["MACD_MACD"] - min(normalizedDf["MACD_MACD"])) / (max(normalizedDf["MACD_MACD"]) - min(normalizedDf["MACD_MACD"]))) - 0.5 ) * 2
        
        
        normalizedOut = normalizedDf["close"] 

        return(cleanedDf, normalizedDf, normalizedOut, featurSetpoints)
    
    
    
    def featureSelection(self):
        
       df = self. normalization()
       selected_features = df[1]

       return(selected_features, df[2], df[0], df[1]["label"], df[3]) 
   
    def dataPreProcessing(self):
        
        df = self. featureSelection()[0].values
        trainData =  df[: -self. testDays, :]
        testData = df[-self. testDays : , :]
        
        return(trainData,testData)
        
    
    def IOSelection(self):
        df1 = self. featureSelection()
       
        inputs__, outputs__ = df1[0], df1[1]
        inputs_ = inputs__[: -self. testDays, :]
        outputs_ = outputs__[: -self. testDays]
        inputs = inputs_[:-self.outputDays, :]
        outputs = outputs_[self.inputDays : ]
        inputTest = inputs__[-(self. testDays+self.inputDays)  : , :]
        outputTest = outputs__[-self. testDays : ]


        xTrain = list()
        yTrain = list()
        xTest = list()
        yTest = list()
        
    
        
        for i in range(self. inputDays, len(inputs)+1):
            xTrain.append(inputs[i-self. inputDays:i, :])
        
        for i in range (self. outputDays, len(outputs)+1):
            yTrain.append(outputs[i - self. outputDays:i])
        
        for i in range(self. inputDays, len(inputTest)):
            xTest.append(inputTest[i-self. inputDays:i, :])
        
        for i in range (self. outputDays, len(outputTest)+1):
            yTest.append(outputTest[i - self. outputDays:i])
        
       
        
        
        
        xTrain, yTrain, xTest, yTest, outputTest = np.array(xTrain), np.array(yTrain), np.array(xTest), np.array(yTest), np.array(outputTest)
        xTrain = np.reshape(xTrain,(xTrain.shape[0], xTrain.shape[1] , xTrain.shape[2]))
        yTrain = np.reshape(yTrain,(yTrain.shape[0], self. outputDays))
        xTest = np.reshape(xTest,(xTest.shape[0], xTest.shape[1] , xTest.shape[2]))
        yTest = np.reshape(yTest,(yTest.shape[0], self. outputDays))
        
        import random 
        j = list(zip(xTrain, yTrain))
        selected = random.sample(j, len(yTrain))
        xTrain = np.array([td for (td, l) in selected])
        yTrain = np.array([l for (td, l) in selected])
        
        
        # import random 
        # j = list(zip(xTest, yTest))
        # selected = random.sample(j, len(yTest))
        # xTest = np.array([td for (td, l) in selected])
        # yTest = np.array([l for (td, l) in selected])
        
        
        proportionOfValidationData = 0.1
        numberOfValidationData = int(len(xTrain) * proportionOfValidationData)

        
        xVal = xTrain[-numberOfValidationData:]
        yVal = yTrain[-numberOfValidationData:]
        xTrain_ = xTrain[0: len(xTrain) - len(xVal)]
        yTrain_ = yTrain[0:len(xTrain) - len(xVal)]
        
        
        return(xTrain_, yTrain_, xTest, yTest, xVal, yVal, df1[4])
    