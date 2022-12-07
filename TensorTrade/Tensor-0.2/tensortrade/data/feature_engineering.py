############ Libs ############
import  numpy as np
import  pandas as pd
from warnings import simplefilter
##############################

class FeatureEngineering:

    def __init__(self, raw_dataset,config):

        self.data_dict = {}
        self.window_sizes= config['window_sizes']
        self.test_percent= config['TEST_DATA_PERCENTAGE']

        dataset= (raw_dataset[["open", "high", "low", "close", "volume",
            "numTrades", "bid_volume", "ask_volume",
            "bid_number", "ask_number",
            "askbidDiffHigh", "askbidDifflow",
            "askbidnumTradesDiffHigh",
            "askbidnumTradesDifflow",
            "UpDownvolDiffHigh", "UpDownvolDifflow",
            "sum_lob_bid", "sum_lob_ask","lob_bid_price",
            "lob_bid_volume","lob_ask_price",
            "lob_ask_volume"]])

        for col in dataset.columns:
            self.data_dict[col] = dataset[col].to_list() if isinstance(dataset[col][0],list) else np.array(dataset[col])
            
        self.data_index = np.array(dataset.index.to_list())
        self.data_length = len(dataset)
        self.params = self.window_sizes # window sizes

        self.last_row_index = {x: -1 for x in self.params}
  

    def add_orderbook_features(self):

        for param in self.params:
            self.data_dict[f'bid_number_{param}min'] = np.array([0.0]*self.data_length)
            self.data_dict[f'ask_number_{param}min'] = np.array([0.0]*self.data_length)
            self.data_dict[f'power_ratio_{param}min'] = np.array([0.0]*self.data_length)
            self.data_dict[f'bid_volume_{param}min'] = np.array([0.0]*self.data_length)
            self.data_dict[f'ask_volume_{param}min'] = np.array([0.0]*self.data_length)
            self.data_dict[f'counter_ratio_{param}min'] = np.array([0.0]*self.data_length)
        for current_index in range(self.data_length):
            for param in set(self.params):
                row_index = self.last_row_index[param] if self.last_row_index[param] != -1 else 0
                while (self.data_index[current_index] - self.data_index[row_index]).seconds / 60 >= param:
                    row_index += 1
                self.last_row_index[param] = row_index
                bid_volume = np.sum(self.data_dict['bid_volume'][self.last_row_index[param]:current_index+1])
                ask_volume = np.sum(self.data_dict['ask_volume'][self.last_row_index[param]:current_index+1])
                bid_number = np.sum(self.data_dict['bid_number'][self.last_row_index[param]:current_index+1])
                ask_number = np.sum(self.data_dict['ask_number'][self.last_row_index[param]:current_index+1])
                self.data_dict[f'bid_volume_{param}min'][current_index] = bid_volume
                self.data_dict[f'ask_volume_{param}min'][current_index] = ask_volume
                self.data_dict[f'bid_number_{param}min'][current_index] = bid_number
                self.data_dict[f'ask_number_{param}min'][current_index] = ask_number
                counter_ratio = bid_volume / ask_volume if bid_volume > ask_volume else -ask_volume / bid_volume
                self.data_dict[f'counter_ratio_{param}min'][current_index] = counter_ratio
                bid_ratio = bid_volume / bid_number
                ask_ratio = ask_volume / ask_number
                power_ratio = bid_ratio / ask_ratio if bid_ratio > ask_ratio else -ask_ratio / bid_ratio
                self.data_dict[f'power_ratio_{param}min'][current_index] = power_ratio
        return self.data_dict

    def zscore_nomalization(self,data, freq=2000):

            data["Muopen"] = data["open"].rolling(window=freq,).mean()
            data["STDopen"] = data["open"].rolling(window=freq,).std()
            #
            data["Muhigh"] = data["high"].rolling(window=freq,).mean()
            data["STDhigh"] = data["high"].rolling(window=freq,).std()
            #
            data["Mulow"] = data["low"].rolling(window=freq,).mean()
            data["STDlow"] = data["low"].rolling(window=freq,).std()
            #
            data["Muclose"] = data["close"].rolling(window=freq,).mean()
            data["STDclose"] = data["close"].rolling(window=freq,).std()
            #
            data["Muvolume"] = data["volume"].rolling(window=freq,).mean()
            data["STDvolume"] = data["volume"].rolling(window=freq,).std()
            #
            data["MunumTrades"] = data["numTrades"].rolling(window=freq,).mean()
            data["STDnumTrades"] = data["numTrades"].rolling(window=freq,).std()
            #
            data["Mubid_volume"] = data["bid_volume"].rolling(window=freq,).mean()
            data["STDbid_volume"] = data["bid_volume"].rolling(window=freq,).std()
            #
            data["Muask_volume"] = data["ask_volume"].rolling(window=freq,).mean()
            data["STDask_volume"] = data["ask_volume"].rolling(window=freq,).std()
            #
            data["Mubid_number"] = data["bid_number"].rolling(window=freq,).mean()
            data["STDbid_number"] = data["bid_number"].rolling(window=freq,).std()
            #
            data["Muask_number"] = data["ask_number"].rolling(window=freq,).mean()
            data["STDask_number"] = data["ask_number"].rolling(window=freq,).std()
            #
            data["Musum_lob_bid"] = data["sum_lob_bid"].rolling(window=freq,).mean()
            data["STDsum_lob_bid"] = data["sum_lob_bid"].rolling(window=freq,).std()
            #
            data["Musum_lob_ask"] = data["sum_lob_ask"].rolling(window=freq,).mean()
            data["STDsum_lob_ask"] = data["sum_lob_ask"].rolling(window=freq,).std()
            #
            data["Mubid_volume_60min"] = data["bid_volume_60min"].rolling(window=freq,).mean()
            data["STDbid_volume_60min"] = data["bid_volume_60min"].rolling(window=freq,).std()
            #
            data["Mupower_ratio_60min"] = data["power_ratio_60min"].rolling(window=freq,).mean()
            data["STDpower_ratio_60min"] = data["power_ratio_60min"].rolling(window=freq,).std()
            #
            data["Muask_number_60min"] = data["ask_number_60min"].rolling(window=freq,).mean()
            data["STDask_number_60min"] = data["ask_number_60min"].rolling(window=freq,).std()
            #
            data["Mubid_number_60min"] = data["bid_number_60min"].rolling(window=freq,).mean()
            data["STDbid_number_60min"] = data["bid_number_60min"].rolling(window=freq,).std()
            #
            data["Mucounter_ratio_20min"] = data["counter_ratio_20min"].rolling(window=freq,).mean()
            data["STDcounter_ratio_20min"] = data["counter_ratio_20min"].rolling(window=freq,).std()
            #
            data["Muask_volume_20min"] = data["ask_volume_20min"].rolling(window=freq,).mean()
            data["STDask_volume_20min"] = data["ask_volume_20min"].rolling(window=freq,).std()
            #
            data["Mubid_volume_20min"] = data["bid_volume_20min"].rolling(window=freq,).mean()
            data["STDbid_volume_20min"] = data["bid_volume_20min"].rolling(window=freq,).std()
            #
            data["Mupower_ratio_20min"] = data["power_ratio_20min"].rolling(window=freq,).mean()
            data["STDpower_ratio_20min"] = data["power_ratio_20min"].rolling(window=freq,).std()
            #
            data["Muask_number_20min"] = data["ask_number_20min"].rolling(window=freq,).mean()
            data["STDask_number_20min"] = data["ask_number_20min"].rolling(window=freq,).std()
            #
            data["Mubid_number_20min"] = data["bid_number_20min"].rolling(window=freq,).mean()
            data["STDbid_number_20min"] = data["bid_number_20min"].rolling(window=freq,).std()
            #
            data["Mucounter_ratio_5min"] = data["counter_ratio_5min"].rolling(window=freq,).mean()
            data["STDcounter_ratio_5min"] = data["counter_ratio_5min"].rolling(window=freq,).std()
            #
            data["Muask_volume_5min"] = data["ask_volume_5min"].rolling(window=freq,).mean()
            data["STDask_volume_5min"] = data["ask_volume_5min"].rolling(window=freq,).std()
            #
            data["Mubid_volume_5min"] = data["bid_volume_5min"].rolling(window=freq,).mean()
            data["STDbid_volume_5min"] = data["bid_volume_5min"].rolling(window=freq,).std()
            #
            data["Mupower_ratio_5min"] = data["power_ratio_5min"].rolling(window=freq,).mean()
            data["STDpower_ratio_5min"] = data["power_ratio_5min"].rolling(window=freq,).std()
            #
            data["Muask_number_5min"] = data["ask_number_5min"].rolling(window=freq,).mean()
            data["STDask_number_5min"] = data["ask_number_5min"].rolling(window=freq,).std()
            #
            data["Mubid_number_5min"] = data["bid_number_5min"].rolling(window=freq,).mean()
            data["STDbid_number_5min"] = data["bid_number_5min"].rolling(window=freq,).std()

            #
            data["Muask_volume_60min"] = data["ask_volume_60min"].rolling(window=freq,).mean()
            data["STDask_volume_60min"] = data["ask_volume_60min"].rolling(window=freq,).std()
            #
            #
            data["Mucounter_ratio_60min"] = data["counter_ratio_60min"].rolling(window=freq,).mean()
            data["STDcounter_ratio_60min"] = data["counter_ratio_60min"].rolling(window=freq,).std()

            ############################
            data["Zscoreopen"] = (data["open"] - data["Muopen"]) / data["STDopen"]
            data["Zscorehigh"] = (data["high"] - data["Muhigh"]) / data["STDhigh"]
            data["Zscorelow"] = (data["low"] - data["Mulow"]) / data["STDlow"]
            data["Zscoreclose"] = (data["close"] - data["Muclose"]) / data["STDclose"]
            data["Zscorevolume"] = (data["volume"] - data["Muvolume"]) / data["STDvolume"]
            data["ZscorenumTrades"] = (data["numTrades"] - data["MunumTrades"]) / data["STDnumTrades"]
            data["Zscorebid_volume"] = (data["bid_volume"] - data["Mubid_volume"]) / data["STDbid_volume"]
            data["Zscoreask_volume"] = (data["ask_volume"] - data["Muask_volume"]) / data["STDask_volume"]
            data["Zscorebid_number"] = (data["bid_number"] - data["Mubid_number"]) / data["STDbid_number"]
            data["Zscoreask_number"] = (data["ask_number"] - data["Muask_number"]) / data["STDask_number"]
            data["Zscoresum_lob_bid"] = (data["sum_lob_bid"] - data["Musum_lob_bid"]) / data["STDsum_lob_bid"]
            data["Zscoresum_lob_ask"] = (data["sum_lob_ask"] - data["Musum_lob_ask"]) / data["STDsum_lob_ask"]
           
           
            data["Zscorebid_number_5min"] = (data["bid_number_5min"] - data["Mubid_number_5min"]) / data["STDbid_number_5min"]
            data["Zscoreask_number_5min"] = (data["ask_number_5min"] - data["Muask_number_5min"]) / data["STDask_number_5min"]
            data["Zscorepower_ratio_5min"] = (data["power_ratio_5min"] - data["Mupower_ratio_5min"]) / data["STDpower_ratio_5min"]
            data["Zscorebid_volume_5min"] = (data["bid_volume_5min"] - data["Mubid_volume_5min"]) / data["STDbid_volume_5min"]
            data["Zscoreask_volume_5min"] = (data["ask_volume_5min"] - data["Muask_volume_5min"]) / data["STDask_volume_5min"]
            data["Zscorecounter_ratio_5min"] = (data["Mucounter_ratio_5min"] - data["counter_ratio_5min"]) / data["STDcounter_ratio_5min"]
            data["Zscorebid_number_20min"] = (data["bid_number_20min"] - data["Mubid_number_20min"]) / data["STDbid_number_20min"]
            data["Zscoreask_number_20min"] = (data["ask_number_20min"] - data["Muask_number_20min"]) / data["STDask_number_20min"]
            data["Zscorepower_ratio_20min"] = (data["power_ratio_20min"] - data["Mupower_ratio_20min"]) / data["STDpower_ratio_20min"]
            data["Zscorebid_volume_20min"] = (data["bid_volume_20min"] - data["Mubid_volume_20min"]) / data["STDbid_volume_20min"]
            data["Zscoreask_volume_20min"] = (data["ask_volume_20min"] - data["Muask_volume_20min"]) / data["STDask_volume_20min"]
            data["Zscorecounter_ratio_20min"] = (data["counter_ratio_20min"] - data["Mucounter_ratio_20min"]) / data["STDcounter_ratio_20min"]
            data["Zscorebid_number_60min"] = (data["bid_number_60min"] - data["Mubid_number_60min"]) / data["STDbid_number_60min"]
            data["Zscoreask_number_60min"] = (data["ask_number_60min"] - data["Muask_number_60min"]) / data["STDask_number_60min"]
            data["Zscorepower_ratio_60min"] = (data["power_ratio_60min"] - data["Mupower_ratio_60min"]) / data["STDpower_ratio_60min"]
            data["Zscorebid_volume_60min"] = (data["bid_volume_60min"] - data["Mubid_volume_60min"]) / data["STDbid_volume_60min"]
            data["Zscoreask_volume_60min"] = (data["ask_volume_60min"] - data["Muask_volume_60min"]) / data["STDask_volume_60min"]
            data["Zscorecounter_ratio_60min"] = (data["counter_ratio_60min"] - data["Mucounter_ratio_60min"]) / data["STDcounter_ratio_60min"]
           
            #
            data.drop(['ZscorenumTrades','open', 'high', 'low', 'close', 'volume', 'numTrades', 'bid_volume','ask_volume', 'bid_number', 'ask_number', 'sum_lob_bid', 'sum_lob_ask','Muopen','STDopen', 'Muhigh', 'STDhigh',\
                'bid_number_5min', 'ask_number_5min', 'power_ratio_5min','bid_volume_5min', 'ask_volume_5min', 'counter_ratio_5min','bid_number_20min', 'ask_number_20min', 'power_ratio_20min',\
                'bid_volume_20min', 'ask_volume_20min', 'counter_ratio_20min', 'bid_number_60min', 'ask_number_60min', 'power_ratio_60min','bid_volume_60min', 'ask_volume_60min', 'counter_ratio_60min',\
                'Mulow', 'STDlow','Muclose', 'STDclose', 'Muvolume', 'STDvolume','MunumTrades','STDnumTrades','Mubid_volume','STDbid_volume','Muask_volume','STDask_volume','Mubid_number',\
                'STDbid_number','Muask_number','STDask_number','Musum_lob_bid','STDsum_lob_bid','Musum_lob_ask','STDsum_lob_ask',\
                'Mucounter_ratio_60min','STDcounter_ratio_60min','Muask_volume_60min','STDask_volume_60min','Muask_number_5min','STDask_number_5min','Mubid_number_5min','STDbid_number_5min','Mubid_volume_5min','STDbid_volume_5min','Mupower_ratio_5min','STDpower_ratio_5min',\
                'Mucounter_ratio_5min','STDcounter_ratio_5min','Muask_volume_5min','STDask_volume_5min','Muask_number_20min','STDask_number_20min',\
                'Mubid_number_20min','STDbid_number_20min','Mubid_volume_20min','STDbid_volume_20min','Mupower_ratio_20min','STDpower_ratio_20min','Mubid_number_60min','STDbid_number_60min','Mucounter_ratio_20min','STDcounter_ratio_20min','Muask_volume_20min','STDask_volume_20min',\
                'Mubid_volume_60min','STDbid_volume_60min','Mupower_ratio_60min','STDpower_ratio_60min','Muask_number_60min','STDask_number_60min'],axis=1,inplace=True)
            
            return data[freq:]
            
    def order_book(self,orderbook_dataset):
            NF=5
            rows=[]
            for i in range(len(orderbook_dataset)):
                try:
                    row =  orderbook_dataset['lob_bid_price'][i].split(',')[0:NF]+\
                    orderbook_dataset['lob_bid_volume'][i].split(',')[0:NF]+\
                    orderbook_dataset['lob_ask_price'][i].split(',')[0:NF]+\
                    orderbook_dataset['lob_ask_volume'][i].split(',')[0:NF]
                    rows.append(row)
                except:
                    pass
            unnormal_data=  pd.DataFrame(rows).astype('float') 
            normal_data = pd.DataFrame(rows).astype('float')
            window_size=2000
            col_mean = normal_data.rolling(window=window_size).mean()
            col_std = normal_data.rolling(window=window_size).std()
            normal_data = (normal_data - col_mean)/col_std
            normal_data = normal_data[window_size:]

            return normal_data,unnormal_data[window_size:]
     

    def add_all_features(self):

        self.add_orderbook_features()
        new_data_set = pd.DataFrame( self.data_dict,index= self.data_index)
        new_data_set.drop(['askbidDiffHigh','askbidDifflow', 'askbidnumTradesDiffHigh', 'askbidnumTradesDifflow',
       'UpDownvolDiffHigh', 'UpDownvolDifflow','lob_bid_price','lob_bid_volume','lob_ask_price','lob_ask_volume'],axis=1,inplace=True)
        
        new_data_set_un = pd.DataFrame( self.data_dict,index= self.data_index)
        new_data_set_un.drop(['askbidDiffHigh','askbidDifflow', 'askbidnumTradesDiffHigh', 'askbidnumTradesDifflow',
       'UpDownvolDiffHigh', 'UpDownvolDifflow','lob_bid_price','lob_bid_volume','lob_ask_price','lob_ask_volume'],axis=1,inplace=True)
       
        orderbook_dataset = pd.DataFrame( self.data_dict,index= self.data_index)
        orderbook_dataset.drop(['askbidDiffHigh','askbidDifflow', 'askbidnumTradesDiffHigh', 'askbidnumTradesDifflow',
       'UpDownvolDiffHigh', 'UpDownvolDifflow'],axis=1,inplace=True)
        
       # Ignore padas warnings
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        unnormal_data = new_data_set_un[2000:]
        Normalize_data=self.zscore_nomalization(new_data_set)
        orderbook_normal,orderbook_unnormal= self.order_book(orderbook_dataset)

        total_dates= list(unnormal_data.index)

        Normalize_data.reset_index(drop=True,inplace=True)
        unnormal_data.reset_index(drop=True,inplace=True)
        orderbook_normal.reset_index(drop=True,inplace=True)
        orderbook_unnormal.reset_index(drop=True,inplace=True)

        total_normal_data= pd.concat([Normalize_data,orderbook_normal],axis=1)  
        total_unnormal_data= pd.concat([unnormal_data,orderbook_unnormal],axis=1)  

        # testNumbers = int(len(total_normal_data) * self.test_percent)

        # trainData =  total_normal_data.iloc[: -testNumbers]
        # testData =   total_normal_data.iloc[-testNumbers : ]

        # trainData_unnormal =  total_unnormal_data.iloc[: -testNumbers]
        # testData_unnormal = total_unnormal_data.iloc[-testNumbers : ]

        # train_dates= total_dates[: -testNumbers]
        # test_dates= total_dates[-testNumbers: ]
        #return normal and unnormal datasets
        return (total_normal_data,total_unnormal_data,total_dates)
