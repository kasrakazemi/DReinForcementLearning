

import pandas as pd

#timeframe=5

# path = "E:/TenSurf/tensurfrl/data/XAUUSD_M30.csv"   #input("please Enter Data Path here:    ")
# data_ = pd.read_csv(path)

# # data_ = data_.resample(f'{timeframe}min').agg( {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})

# path = "E:/TenSurf/tensurfrl/data/XAUUSD_M30.csv"   #input("please Enter Data Path here:    ")
# data = pd.read_csv(path)
# data = data.resample(f'{timeframe}min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})


actions = ["Buy", "Sell", "Hold"]     # we can change it in future
actionConverting = [1, -1, 0]

sequenceNumber = 1   # we can change it in future
#testNumbers = int(len(data) * 0.1)

outPutOfQ = ["power of Buy", "powe of Sell", "power of Hold"]

numberOfFirstBatchData = 64

margin = 1000

relayMemorySize = 10000000

agentName = "a.h"

epsilon = 1.0
epsilonFinal = 0.01
epsilonDecay = 0.95

gamma = 0.95

batchSize = 64

Sl = 0.1
Tp = 0.2
levrage = 10

epsilonTest = 0.1

NqPath = "C:/Users/kasra/Downloads/Ten-Surf/Tensurf-RL/tensurfrl/data/LOB_NQU22-CME_2_1_10_20days.lob"





