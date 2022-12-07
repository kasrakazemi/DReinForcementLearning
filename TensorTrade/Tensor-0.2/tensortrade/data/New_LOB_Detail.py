##################################################### joda kardan lob levels
NF=10
rows=[]
for i in range(len(df)):
    try:
        row =  df['LOB_BidPrices'][i].split(',')[0:NF]+\
        df['LOB_BidVolumes'][i].split(',')[0:NF]+\
        df['LOB_AskPrices'][i].split(',')[0:NF]+\
        df['LOB_AskVolumes'][i].split(',')[0:NF]
        rows.append(row)
    except:
        print(i)
        pass

input = pd.DataFrame(rows).astype('float')
input
################################################################### normalization
window_size=2000
col_mean = input.rolling(window=window_size).mean()
col_std = input.rolling(window=window_size).std()
input = (input - col_mean)/col_std
input = input[window_size:]
labels = labels.iloc[window_size:]
################################################################## train test validation split
split_train_val = int(np.floor(len(dataX) * 0.6))
trainX = dataX[:split_train_val]
trainY = dataY[:split_train_val]
split_val_test = int(np.floor(len(dataX) * 0.70))
valX = dataX[split_train_val:split_val_test]
valY = dataY[split_train_val:split_val_test]
testX = dataX[split_val_test:]
testY = dataY[split_val_test:]


data =input
lookback_cnn = 50
level_data_num=40
N = data.shape[0]
dataX = np.zeros((N - lookback_cnn + 1, lookback_cnn, level_data_num))
for i in range(dataX.shape[0]):
    tmp_sample = np.zeros((lookback_cnn, level_data_num))
    for j in range(lookback_cnn):
        ind = i + j
        tmp_sample[j, :] = data.iloc[ind]
    dataX[i, :, :] = tmp_sample
dataX=dataX.reshape((dataX.shape[0], lookback_cnn, level_data_num,1))
labels = labels.iloc[lookback_cnn-1:]
#####################################################################



