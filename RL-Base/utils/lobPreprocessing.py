
import pandas as pd
import ast
import numpy as np


class lobPreprocessing:
    
    def __init__(self, pathForLOB, lobSwitch = True, mobSwitch = False):
        
        self. pathForLOB = pathForLOB
        
        
        if lobSwitch:
            self. col_names = ["DateTime","open","high","low","close","Volume","NumTrades","BidVolume","AskVolume","SumBid","SumAsk","BidPrices","BidVolumes","AskPrices","AskVolumes"]
            self. col_list = ["BidPrices","BidVolumes","AskPrices","AskVolumes"]
        
        if mobSwitch:
            self. col_names = ["DateTime","Open","High","Low","Last","Volume","NumTrades","BidVolume","AskVolume",\
                "SumBidVolume","SumBidNumTrades","SumAskVolume","SumAskNumTrades",\
                    "BidPrices","BidVolumes","BidNumTrades","AskPrices","AskVolumes","AskNumTrades"]
            self. col_list =["BidPrices","BidVolumes","BidNumTrades","AskPrices","AskVolumes","AskNumTrades"]
        
        
    def string_to_nplist(self, x):
       if pd.isnull(x):
           return []
       else:
           return np.array(ast.literal_eval(x))

    def read_data(self, file_name, col_names, col_list):
       data = pd.read_csv(file_name, names=col_names, delimiter="|")
       for col in col_list:
           data[col] = data[col].apply(lambda x: self. string_to_nplist(x))
       data["DateTime"] = pd.to_datetime(data["DateTime"])
       data.set_index(["DateTime"], inplace=True)
       return data

    def clean_lob(self, data, weight_mid_price=0.5, cols_need=["BidPrices","BidVolumes","AskPrices","AskVolumes"], num_level=10):
       lst_valid_samples = []
       mid_prices = []
       for ind, row in data.iterrows():
           if len(row["BidPrices"]) and len(row["AskPrices"]):
               if (row["BidPrices"].shape[0] == num_level) and (row["AskPrices"].shape[0] == num_level):
                   lst_valid_samples.append(ind)
                   mid_p = weight_mid_price * row["BidPrices"][0] + (1 - weight_mid_price) * row["AskPrices"][0]
                   mid_prices.append(mid_p)
       ret_data = pd.DataFrame(index=lst_valid_samples, data=data.loc[lst_valid_samples, cols_need])
       ret_data["Midprice"] = mid_prices
       return ret_data

    def func_cc(self, x):
       ret = np.concatenate((x.ZscoreAskPrices, x.ZscoreAskVolumes, x.ZscoreBidPrices, x.ZscoreBidVolumes))
       return ret
       
    def zscore_nomalization(self, data, freq="5D", min_periods=4*24*60):
       data["AvgBidPrices"] = data["BidPrices"].apply(lambda x: np.mean(x))
       data["AvgBidVolumes"] = data["BidVolumes"].apply(lambda x: np.mean(x))
       data["AvgAskPrices"] = data["AskPrices"].apply(lambda x: np.mean(x))
       data["AvgAskVolumes"] = data["AskVolumes"].apply(lambda x: np.mean(x))
       data["MuBidPrice"] = data["AvgBidPrices"].rolling(window=freq, min_periods=min_periods).mean()
       data["STDBidPrice"] = data["AvgBidPrices"].rolling(window=freq, min_periods=min_periods).std()
       data["MuBidVolume"] = data["AvgBidVolumes"].rolling(window=freq, min_periods=min_periods).mean()
       data["STDBidVolume"] = data["AvgBidVolumes"].rolling(window=freq, min_periods=min_periods).std()
       data["MuAskPrice"] = data["AvgAskPrices"].rolling(window=freq, min_periods=min_periods).mean()
       data["STDAskPrice"] = data["AvgAskPrices"].rolling(window=freq, min_periods=min_periods).std()
       data["MuAskVolume"] = data["AvgAskVolumes"].rolling(window=freq, min_periods=min_periods).mean()
       data["STDAskVolume"] = data["AvgAskVolumes"].rolling(window=freq, min_periods=min_periods).std()
       data["ZscoreBidPrices"] = (data["BidPrices"] - data["MuBidPrice"]) / data["STDBidPrice"]
       data["ZscoreBidVolumes"] = (data["BidVolumes"] - data["MuBidVolume"]) / data["STDBidVolume"]
       data["ZscoreAskPrices"] = (data["AskPrices"] - data["MuAskPrice"]) / data["STDAskPrice"]
       data["ZscoreAskVolumes"] = (data["AskVolumes"] - data["MuAskVolume"]) / data["STDAskVolume"]
       data["ConcatLOB"] =  data[["ZscoreAskPrices", "ZscoreAskVolumes", "ZscoreBidPrices", "ZscoreBidVolumes"]].apply(lambda x: self. func_cc(x), axis=1)
       

     
    def final(self):
        a = self. read_data(self. pathForLOB, self. col_names,  self. col_list)

        b = self. clean_lob(a)

        self. zscore_nomalization(b)

        b.dropna(how='any', axis=0, inplace=True)
        
        c = b["ConcatLOB"]
        x  = np.stack(c)
        testNumbers = int(len(x) * 0.1)
        trainData =  x[: -testNumbers, :]
        testData = x[-testNumbers : , :]
        
        
        realData = a.iloc[ :, 0:4]
        realData.iloc[len(a) - len(x) : , ]
        realData.reset_index(drop=True, inplace=True)
        
        dataTrain = realData.iloc[ :-testNumbers , : ]
        dataTrain.reset_index(drop=True, inplace=True)
        dataTest = dataTrain.iloc[-testNumbers : , : ]
        dataTest.reset_index(drop=True, inplace=True) 


      
        return(trainData, testData, dataTrain, dataTest)

    
    
    
    
    