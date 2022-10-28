import shutil
import ntpath
import os
import numpy as np
import math
from tensorflow import keras
####################################

def k_moving_average(arr, window_size=20):
    moving_averages = []
    i = 0
    while i < len(arr) - window_size + 1:
        window_average = round(np.sum(arr[i:i+window_size]) / window_size, 2)
        moving_averages.append(window_average)
        i += 1
    return moving_averages
def process_data(path, ind, n, first_line, cols, dirname, level=10, k=[5,10,20,50,100], look_back=50):
    data = [first_line]
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= ind[0] and i < ind[1]:
                data.append(line)
    ltps = []
    for line in data[1:]:
        splits = line.split('|')
        ltp = float(splits[cols['Candle_LastTradePrice']])
        ltps.append(ltp)
    multi_labels = []
    for h in k:
        ma = k_moving_average(ltps, h)
        k_minus = np.array([np.nan] * (h-1) + ma)
        k_plus = np.array(ma + [np.nan] * (h-1))
        smoothing = (k_plus - k_minus) / k_minus
        alpha = np.std(smoothing[~np.isnan(smoothing)]) / 2
        labels = []
        for s in smoothing[~np.isnan(smoothing)]:
            if s > alpha:
                labels.append(s)
            elif s < -alpha:
                labels.append(s)
            else:
                labels.append(s)
        labels_str = [np.nan] * (h-1) + labels + [np.nan] * (h-1)
        labels_str = [str(i) for i in labels_str]
        multi_labels.append(labels_str)
    data[0] = data[0][:-1] + '|Label1|Label2|Label3|Label4|Label5\n'
    for i in range(len(data)):
        if i == 0:
          continue
        else:
          splits = data[i].split('|')
          bidprices = splits[cols['LOB_BidPrices']].split(',')
          bidprices = [p for p in bidprices if p != ''][-level:]
          askprices = splits[cols['LOB_AskPrices']].split(',')
          askprices = [p for p in askprices if p != ''][:level]
          bidvolumes = splits[cols['LOB_BidVolumes']].split(',')
          bidvolumes = [v for v in bidvolumes if v != ''][-level:]
          askvolumes = splits[cols['LOB_AskVolumes']].split(',')
          askvolumes = [v for v in askvolumes if v != ''][:level]
          splits[cols['LOB_BidPrices']] = ','.join(bidprices)
          splits[cols['LOB_AskPrices']] = ','.join(askprices)
          splits[cols['LOB_BidVolumes']] = ','.join(bidvolumes)
          splits[cols['LOB_AskVolumes']] = ','.join(askvolumes)
          splits[-1] = splits[-1][:-1]
          splits.append(multi_labels[0][i-1])
          splits.append(multi_labels[1][i-1])
          splits.append(multi_labels[2][i-1])
          splits.append(multi_labels[3][i-1])
          splits.append(multi_labels[4][i-1] + '\n')
          data[i] = '|'.join(splits)
    filename = ntpath.basename(path).split('.')[0] + 'labeled{}.csv'.format(n)
    with open(os.path.join(dirname, 'labeled', filename), 'w') as f:
        for line in data:
            f.write(line)
    corrupts = []
    for i, line in enumerate(data):
        splits = line.split('|')
        bidprices = splits[cols['LOB_BidPrices']].split(',')
        askprices = splits[cols['LOB_AskPrices']].split(',')
        bidvolumes = splits[cols['LOB_BidVolumes']].split(',')
        askvolumes = splits[cols['LOB_AskVolumes']].split(',')
        if len(bidprices) < level or len(askprices) < level or '' in askprices or '' in bidprices:    # corrupt condition
            corrupts.append(i)
    print(len(corrupts))
    subfiles = []
    for i in range(len(corrupts) - 1):
        start = corrupts[i]
        end = corrupts[i+1]
        if end - start - 1 >= look_back:
            subfiles.append(data[start+1:end])
    if (len(data) - 1 - corrupts[-1]) >= look_back:
        subfiles.append(data[corrupts[-1]+1:])
    total = []
    for f in subfiles:
        total += f
    filename = ntpath.basename(path).split('.')[0] + '_clean_labeled_{}.csv'.format(n)
    with open(os.path.join(dirname, 'clean_labeled', filename), 'w') as f:
        for line in total:
            f.write(line)
    samplesX = []
    samplesY = []
    for f in subfiles:
        for i in range(look_back-1, len(f)):
            splits = f[i].split('|')
            sample_labels = [float(splits[-5])+1, float(splits[-4])+1, float(splits[-3])+1, float(splits[-2])+1, float(splits[-1][:-1])+1]
            if any([math.isnan(i) for i in sample_labels]):
                continue
            total_ask_prices = []
            total_ask_volumes = []
            total_bid_prices = []
            total_bid_volumes = []
            for j in range(i+1-look_back, i+1):
                spltis = f[j].split('|')
                total_ask_prices = total_ask_prices + splits[cols['LOB_AskPrices']].split(',')
                total_bid_prices = total_bid_prices + splits[cols['LOB_BidPrices']].split(',')
                if '' in total_bid_prices:
                    print(splits[cols['LOB_BidPrices']])
                    return splits[cols['LOB_BidPrices']]
                total_ask_volumes = total_ask_volumes + splits[cols['LOB_AskVolumes']].split(',')
                total_bid_volumes = total_bid_volumes + splits[cols['LOB_BidVolumes']].split(',')
            total_ask_prices = np.array([float(i) for i in total_ask_prices])
            total_ask_volumes = np.array([float(i) for i in total_ask_volumes])
            total_bid_prices = np.array([float(i) for i in total_bid_prices])
            total_bid_volumes = np.array([float(i) for i in total_bid_volumes])
            price_mean = np.concatenate([total_ask_prices, total_bid_prices]).mean()
            price_std = np.concatenate([total_ask_prices, total_bid_prices]).std()
            volume_mean = np.concatenate([total_ask_volumes, total_bid_volumes]).mean()
            volume_std = np.concatenate([total_ask_volumes, total_bid_volumes]).std()
            total_ask_prices = (total_ask_prices - price_mean) / price_std
            total_ask_volumes = (total_ask_volumes - volume_mean) / volume_std
            total_bid_prices = (total_bid_prices - price_mean) / price_std
            total_bid_volumes = (total_bid_volumes - volume_mean) / volume_std
            total = np.stack([total_ask_prices, total_bid_prices, total_ask_volumes, total_bid_volumes]).T
            total = total.reshape((look_back, -1))
            samplesX.append(total)
            # samplesY.append(keras.utils.to_categorical(sample_labels, 3))
            samplesY.append(sample_labels)
    X_name = ntpath.basename(path).split('.')[0] + 'X{}.npy'.format(n)
    y_name = ntpath.basename(path).split('.')[0] + 'y{}.npy'.format(n)
    with open(os.path.join(dirname, 'numpy', X_name), 'wb') as f:
        np.save(f, np.array(samplesX).reshape(-1, look_back, level*4, 1))
    with open(os.path.join(dirname, 'numpy', y_name), 'wb') as f:
        np.save(f, np.array(samplesY))
def prepare_data_2(path, level=10, k=[5,10,20,50,100], look_back=50):
    dirname = ntpath.basename(path).split('.')[0] + '_processed'
    count = 0
    with open(path, 'r') as f:
        for line in f:
            if count == 0:
                cols = line.split('|')
                cols[-1] = cols[-1][:-1]
                cols = {cols[i]:i for i in range(len(cols))}
                first_line = line
            count += 1
    n = (count // 200000) + 1
    inds = []
    for i in range(n):
        if (count - i*200000) > 200000:
            if i == 0:
                inds.append((1, (i+1)*200000))
            else:
                inds.append((i*200000, (i+1)*200000))
        else:
            inds.append((i*200000, count))
    try:
        shutil.rmtree(dirname)
        print('Removing old directory and creating a new one')
    except:
        print('No such directory')
        print('Creating new directory')
    os.mkdir(dirname)
    os.mkdir(os.path.join(dirname, 'labeled'))
    os.mkdir(os.path.join(dirname, 'clean_labeled'))
    os.mkdir(os.path.join(dirname, 'numpy'))
    for i, ind in enumerate(inds):
        process_data(path=path, ind=ind, n=i, first_line=first_line, cols=cols, dirname=dirname, level=level, k=k, look_back=look_back)

#######################################################

# !wget -O data.rar https://www.dropbox.com/sh/7unwg9pomo64ih7/AADdJESacvWKGQEDf0OgeUzAa/LOB1_NQU22-CME_10sec_200Levels_NoOverlap.rar?dl=0
# !unrar x '/content/data.rar'
# prepare_data_2('./LOB1_NQU22-CME_10sec_200Levels_NoOverlap.csv')