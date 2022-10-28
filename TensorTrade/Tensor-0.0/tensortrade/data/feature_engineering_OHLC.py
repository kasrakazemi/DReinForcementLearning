############ Libs ############
import  numpy as np
import  pandas as pd
from stockstats import StockDataFrame
##############################

class FeatureEngineering_ohlc:

    def __init__(self, data_path):

        self.data_path= data_path
    #     col_names =['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    #    'NumberOfTrades', 'BidVolume', 'AskVolume']
        self.dataset = pd.read_csv(self.data_path, delimiter=",")
        self.dataset["time"] = (self.dataset["Date"]+self.dataset['Time'])

        #self.dataset.set_index(["time"], inplace=True)
        self.dataset.drop(['Date','Time'],axis=1,inplace=True)
        self.dataset.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"},inplace=True)
        
        self.new_dataset=  pd.DataFrame(self.dataset.loc[1500000:],columns=['time','open','high','low','close','Volume','NumberOfTrades','BidVolume','AskVolume'])
        self.new_dataset.reset_index(drop=True,inplace=True)
        self.new_dataset['rsi_14']= StockDataFrame.retype(self.new_dataset.copy())['rsi']
        self.new_dataset['boll']= StockDataFrame.retype(self.new_dataset.copy())['boll']
        self.new_dataset['boll_ub']= StockDataFrame.retype(self.new_dataset.copy())['boll_ub']
        self.new_dataset['boll_lb']= StockDataFrame.retype(self.new_dataset.copy())['boll_lb']
        self.new_dataset['macd']= StockDataFrame.retype(self.new_dataset.copy())['macd']
        self.new_dataset['adx']= StockDataFrame.retype(self.new_dataset.copy())['adx']
        self.new_dataset.drop(self.new_dataset.index[0:10],inplace=True)

    def add_all_features(self):

        testNumbers = int(len(self.new_dataset) * 0.3)
        trainData =  self.new_dataset.iloc[: -testNumbers]
        testData = self.new_dataset.iloc[-testNumbers : ]

        # return normal and unnormal datasets
        return (trainData,testData)

