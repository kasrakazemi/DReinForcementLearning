

#timeframe=5

# path = "E:/TenSurf/tensurfrl/data/XAUUSD_M30.csv"   #input("please Enter Data Path here:    ")
# data_ = pd.read_csv(path)

# # data_ = data_.resample(f'{timeframe}min').agg( {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})

# path = "E:/TenSurf/tensurfrl/data/XAUUSD_M30.csv"   #input("please Enter Data Path here:    ")
# data = pd.read_csv(path)
# data = data.resample(f'{timeframe}min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})


actions = ["Buy", "Sell", "Hold"]     # we can change it in future

actionConverting = [0,1,2]

sequenceNumber = 1   # we can change it in future
#testNumbers = int(len(data) * 0.1)

outPutOfQ = ["power of Buy", "powe of Sell", "power of Hold"]

batchSize = 32

margin = 1000

relayMemorySize = 10000000


epsilon = 1.0
epsilonFinal = 0.01
epsilonDecay = 0.95

gamma = 0.9



Sl = 0.1
Tp = 0.2
#levrage = 10

#epsilonTest = 0.1

NqPath = "C:/Users/kasra/Downloads/Ten-Surf/Tensurf-RL/tensurfrl/data/LOB1, NQ, 1_1min.csv"





