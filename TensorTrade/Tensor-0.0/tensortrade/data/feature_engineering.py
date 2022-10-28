############ Libs ############
import  numpy as np
import  pandas as pd
from warnings import simplefilter
##############################

class FeatureEngineering:

    def __init__(self, data_path,window_sizes,orderbook_percents,test_percent,normalization_mode):

        self.data_dict = {}
        self.data_path= data_path
        # col_names =["time", "open", "high", "low", "close", "volume",
        #     "numTrades", "bid_volume", "ask_volume",
        #     "bid_number", "ask_number",
        #     "askbidDiffHigh", "askbidDifflow",
        #     "askbidnumTradesDiffHigh",
        #     "askbidnumTradesDifflow",
        #     "UpDownvolDiffHigh", "UpDownvolDifflow",
        #     "sum_lob_bid", "sum_lob_ask","lob_bid_price",
        #     "lob_bid_volume","lob_ask_price",
        #     "lob_ask_volume"]

        col_names=['DateTime','open', 'high', 'low',
       'close', 'Candle_LastTradePrice', 'volume',
       'bid_volume', 'ask_volume',
       'numTrades', 'bid_number',
       'ask_number', 'askbidDiffHigh',
       'askbidDifflow', 'askbidnumTradesDiffHigh',
       'askbidnumTradesDifflow', 'UpDownvolDiffHigh',
       'UpDownvolDifflow', 'ATR', 'sum_lob_bid', 'sum_lob_ask',
       'LOB_SumBidTick', 'LOB_SumAskTick', 'lob_bid_price', 'lob_bid_volume',
       'lob_ask_price', 'lob_ask_volume', 'VAP_Prices', 'VAP_Volumes',
       'VAP_AskVolumes', 'VAP_BidVolumes', 'VAP_NumberOfTrades',
       'VAP_TotalVolume']

        raw_dataset = pd.read_csv(self.data_path, names=col_names,header=0, delimiter="|")
        #raw_dataset.drop([0],inplace=True)
        raw_dataset["DateTime"] = pd.to_datetime(raw_dataset["DateTime"])
        raw_dataset.set_index(["DateTime"], inplace=True)

        dataset= (raw_dataset[["open", "high", "low", "close", "volume",
            "numTrades", "bid_volume", "ask_volume",
            "bid_number", "ask_number",
            "askbidDiffHigh", "askbidDifflow",
            "askbidnumTradesDiffHigh",
            "askbidnumTradesDifflow",
            "UpDownvolDiffHigh", "UpDownvolDifflow",
            "sum_lob_bid", "sum_lob_ask","lob_bid_price",
            "lob_bid_volume","lob_ask_price",
            "lob_ask_volume"]])[:30000] # This is temporary

        for col in dataset.columns:
            self.data_dict[col] = dataset[col].to_list() if isinstance(dataset[col][0],list) else np.array(dataset[col])
        self.data_index = np.array(dataset.index.to_list())
        self.data_length = len(dataset)
        self.params = window_sizes # window sizes
        self.lob_params = orderbook_percents # Percent of order book
        self.last_row_index = {x: -1 for x in self.params}
        self.test_percent = test_percent
        self.nomalization_mode= normalization_mode

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
  
    def add_lob_features(self):

        for param in self.lob_params:
            self.data_dict[f'orderbook{param}'] = np.array([0.0] * self.data_length)
            for col in ['ask', 'bid']:
                self.data_dict[f'sum_lob_{col}_{param}'] = np.array([0.0]*self.data_length)

        for current_index in range(self.data_length):
            close_price = self.data_dict[f'close'][current_index]
            for param in self.lob_params:
                for col in ['ask', 'bid']:
                    for order, volume in zip(self.data_dict[f'lob_{col}_price'][current_index].split(',')[:-1],
                                             self.data_dict[f'lob_{col}_volume'][current_index].split(',')[:-1]):
                        if abs(float(order) - close_price) < param * close_price:
                            self.data_dict[f'sum_lob_{col}_{param}'][current_index] +=int(volume)
            sum_bid = self.data_dict[f'sum_lob_bid_{param}'][current_index]
            sum_ask = self.data_dict[f'sum_lob_ask_{param}'][current_index]
            if sum_bid > sum_ask:
                self.data_dict[f'orderbook{param}'][current_index] = \
                    sum_bid / sum_ask if sum_ask!=0 else sum_bid
            elif sum_ask > sum_bid:
                self.data_dict[f'orderbook{param}'][current_index] = \
                    -sum_ask / sum_bid if sum_bid!= 0 else sum_ask
            else:
                self.data_dict[f'orderbook{param}'][current_index] = 0

        return self.data_dict
    
    def zscore_nomalization(self,data, freq="1D", min_periods=24*30):

        if self.nomalization_mode == 'ZScore':

            data["Muopen"] = data["open"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDopen"] = data["open"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muhigh"] = data["high"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDhigh"] = data["high"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mulow"] = data["low"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDlow"] = data["low"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muclose"] = data["close"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDclose"] = data["close"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muvolume"] = data["volume"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDvolume"] = data["volume"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["MunumTrades"] = data["numTrades"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDnumTrades"] = data["numTrades"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_volume"] = data["bid_volume"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_volume"] = data["bid_volume"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muask_volume"] = data["ask_volume"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_volume"] = data["ask_volume"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_number"] = data["bid_number"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_number"] = data["bid_number"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muask_number"] = data["ask_number"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_number"] = data["ask_number"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_bid"] = data["sum_lob_bid"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_bid"] = data["sum_lob_bid"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_ask"] = data["sum_lob_ask"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_ask"] = data["sum_lob_ask"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muorderbook0.01"] = data["orderbook0.01"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDorderbook0.01"] = data["orderbook0.01"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_ask_0.01"] = data["sum_lob_ask_0.01"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_ask_0.01"] = data["sum_lob_ask_0.01"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_bid_0.01"] = data["sum_lob_bid_0.01"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_bid_0.01"] = data["sum_lob_bid_0.01"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muorderbook0.02"] = data["orderbook0.02"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDorderbook0.02"] = data["orderbook0.02"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_ask_0.02"] = data["sum_lob_ask_0.02"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_ask_0.02"] = data["sum_lob_ask_0.02"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_volume_60min"] = data["bid_volume_60min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_volume_60min"] = data["bid_volume_60min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mupower_ratio_60min"] = data["power_ratio_60min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDpower_ratio_60min"] = data["power_ratio_60min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muask_number_60min"] = data["ask_number_60min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_number_60min"] = data["ask_number_60min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_number_60min"] = data["bid_number_60min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_number_60min"] = data["bid_number_60min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mucounter_ratio_20min"] = data["counter_ratio_20min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDcounter_ratio_20min"] = data["counter_ratio_20min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muask_volume_20min"] = data["ask_volume_20min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_volume_20min"] = data["ask_volume_20min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_volume_20min"] = data["bid_volume_20min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_volume_20min"] = data["bid_volume_20min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mupower_ratio_20min"] = data["power_ratio_20min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDpower_ratio_20min"] = data["power_ratio_20min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muask_number_20min"] = data["ask_number_20min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_number_20min"] = data["ask_number_20min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_number_20min"] = data["bid_number_20min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_number_20min"] = data["bid_number_20min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mucounter_ratio_5min"] = data["counter_ratio_5min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDcounter_ratio_5min"] = data["counter_ratio_5min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muask_volume_5min"] = data["ask_volume_5min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_volume_5min"] = data["ask_volume_5min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_volume_5min"] = data["bid_volume_5min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_volume_5min"] = data["bid_volume_5min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mupower_ratio_5min"] = data["power_ratio_5min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDpower_ratio_5min"] = data["power_ratio_5min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muask_number_5min"] = data["ask_number_5min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_number_5min"] = data["ask_number_5min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Mubid_number_5min"] = data["bid_number_5min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDbid_number_5min"] = data["bid_number_5min"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_bid_0.005"] = data["sum_lob_bid_0.005"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_bid_0.005"] = data["sum_lob_bid_0.005"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_ask_0.005"] = data["sum_lob_ask_0.005"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_ask_0.005"] = data["sum_lob_ask_0.005"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Muorderbook0.005"] = data["orderbook0.005"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDorderbook0.005"] = data["orderbook0.005"].rolling(window=freq, min_periods=min_periods).std()
            #
            data["Musum_lob_bid_0.02"] = data["sum_lob_bid_0.02"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDsum_lob_bid_0.02"] = data["sum_lob_bid_0.02"].rolling(window=freq, min_periods=min_periods).std()
            #
            #
            data["Muask_volume_60min"] = data["ask_volume_60min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDask_volume_60min"] = data["ask_volume_60min"].rolling(window=freq, min_periods=min_periods).std()
            #
            #
            data["Mucounter_ratio_60min"] = data["counter_ratio_60min"].rolling(window=freq, min_periods=min_periods).mean()
            data["STDcounter_ratio_60min"] = data["counter_ratio_60min"].rolling(window=freq, min_periods=min_periods).std()

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
            data["Zscoreorderbook0.01"] = (data["orderbook0.01"] - data["Muorderbook0.01"]) / data["STDorderbook0.01"]
            data["Zscoresum_lob_ask_0.01"] = (data["sum_lob_ask_0.01"] - data["Musum_lob_ask_0.01"]) / data["STDsum_lob_ask_0.01"]
            data["Zscoresum_lob_bid_0.01"] = (data["sum_lob_bid_0.01"] - data["Musum_lob_bid_0.01"]) / data["STDsum_lob_bid_0.01"]
            data["Zscoreorderbook0.02"] = (data["orderbook0.02"] - data["Muorderbook0.02"]) / data["STDorderbook0.02"]
            data["Zscoresum_lob_ask_0.02"] = (data["sum_lob_ask_0.02"] - data["Musum_lob_ask_0.02"]) / data["STDsum_lob_ask_0.02"]
            data["Zscoresum_lob_bid_0.02"] = (data["sum_lob_bid_0.02"] - data["Musum_lob_bid_0.02"]) / data["STDsum_lob_bid_0.02"]
            data["Zscoreorderbook0.005"] = (data["orderbook0.005"] - data["Muorderbook0.005"]) / data["STDorderbook0.005"]
            data["Zscoresum_lob_ask_0.005"] = (data["sum_lob_ask_0.005"] - data["Musum_lob_ask_0.005"]) / data["STDsum_lob_ask_0.005"]
            data["Zscoresum_lob_bid_0.005"] = (data["sum_lob_bid_0.005"] - data["Musum_lob_bid_0.005"]) / data["STDsum_lob_bid_0.005"]
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
                'orderbook0.01', 'sum_lob_ask_0.01', 'sum_lob_bid_0.01','orderbook0.02', 'sum_lob_ask_0.02', 'sum_lob_bid_0.02', 'orderbook0.005', 'sum_lob_ask_0.005', 'sum_lob_bid_0.005',\
                'bid_number_5min', 'ask_number_5min', 'power_ratio_5min','bid_volume_5min', 'ask_volume_5min', 'counter_ratio_5min','bid_number_20min', 'ask_number_20min', 'power_ratio_20min',\
                'bid_volume_20min', 'ask_volume_20min', 'counter_ratio_20min', 'bid_number_60min', 'ask_number_60min', 'power_ratio_60min','bid_volume_60min', 'ask_volume_60min', 'counter_ratio_60min',\
                'Musum_lob_bid_0.005', 'STDsum_lob_bid_0.005', 'Muorderbook0.005','STDorderbook0.005',\
                'Mulow', 'STDlow','Muclose', 'STDclose', 'Muvolume', 'STDvolume','MunumTrades','STDnumTrades','Mubid_volume','STDbid_volume','Muask_volume','STDask_volume','Mubid_number',\
                'STDbid_number','Muask_number','STDask_number','Musum_lob_bid','STDsum_lob_bid','Musum_lob_ask','STDsum_lob_ask','Muorderbook0.01','STDorderbook0.01',\
                'Musum_lob_ask_0.01','STDsum_lob_ask_0.01','Musum_lob_bid_0.01','STDsum_lob_bid_0.01','Muorderbook0.02','STDorderbook0.02','Musum_lob_ask_0.02','STDsum_lob_ask_0.02',\
                'Mucounter_ratio_60min','STDcounter_ratio_60min','Muask_volume_60min','STDask_volume_60min','Musum_lob_ask_0.005','STDsum_lob_ask_0.005','Musum_lob_bid_0.02','STDsum_lob_bid_0.02','Muask_number_5min','STDask_number_5min','Mubid_number_5min','STDbid_number_5min','Mubid_volume_5min','STDbid_volume_5min','Mupower_ratio_5min','STDpower_ratio_5min',\
                'Mucounter_ratio_5min','STDcounter_ratio_5min','Muask_volume_5min','STDask_volume_5min','Muask_number_20min','STDask_number_20min',\
                'Mubid_number_20min','STDbid_number_20min','Mubid_volume_20min','STDbid_volume_20min','Mupower_ratio_20min','STDpower_ratio_20min','Mubid_number_60min','STDbid_number_60min','Mucounter_ratio_20min','STDcounter_ratio_20min','Muask_volume_20min','STDask_volume_20min',\
                'Mubid_volume_60min','STDbid_volume_60min','Mupower_ratio_60min','STDpower_ratio_60min','Muask_number_60min','STDask_number_60min'],axis=1,inplace=True)

            # orderbooks all are nan so we should drop them
            data.drop(['Zscoreorderbook0.01','Zscoreorderbook0.02'],axis=1,inplace= True)
            # nan values should be drop from dataset
            data.dropna(axis=0,inplace=True)

            return data
            
        else:
            pass


    def add_all_features(self):

        self.add_lob_features()
        self.add_orderbook_features()
        new_data_set = pd.DataFrame( self.data_dict,index= self.data_index)
        new_data_set.drop(['askbidDiffHigh','askbidDifflow', 'askbidnumTradesDiffHigh', 'askbidnumTradesDifflow',
       'UpDownvolDiffHigh', 'UpDownvolDifflow','lob_bid_price', 'lob_bid_volume', 'lob_ask_price', 'lob_ask_volume'],axis=1,inplace=True)
        new_data_set_Unnormal = pd.DataFrame( self.data_dict,index= self.data_index)
        new_data_set_Unnormal.drop([ 'numTrades','askbidDiffHigh','askbidDifflow', 'askbidnumTradesDiffHigh', 'askbidnumTradesDifflow',
       'UpDownvolDiffHigh', 'UpDownvolDifflow','lob_bid_price', 'lob_bid_volume', 'lob_ask_price', 'lob_ask_volume','orderbook0.01','orderbook0.02'],axis=1,inplace=True)
       # Ignore padas warnings
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        Normalize_data=self.zscore_nomalization(new_data_set)
        testNumbers = int(len(Normalize_data) * self.test_percent)
        trainData =  Normalize_data.iloc[: -testNumbers]
        testData = Normalize_data.iloc[-testNumbers : ]

        # start_date= str((Normalize_data.index)[0])
        # mask = (new_data_set_Unnormal.index)>= start_date
        mask = list(Normalize_data.index)
        new_data_set_Unnormal=new_data_set_Unnormal.loc[mask]
        trainData_unnormal =  new_data_set_Unnormal.iloc[: -testNumbers]
        testData_unnormal = new_data_set_Unnormal.iloc[-testNumbers : ]
        #return normal and unnormal datasets
        return (trainData_unnormal,testData_unnormal,trainData,testData)

