# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:04:26 2022

@author: a.h
"""


################# imports #############

from utils import *

################ data preprocessing  for forex data ###########
# preProcessedData = dataAnalyse(data, sequenceNumber, testNumbers)
# trainData, testData = preProcessedData.dataPreProcessing()

# differenceBetweenNormalDataAndActualData = len(data_) - (len(trainData) + len(testData))

# dataTrain = data_.iloc[ differenceBetweenNormalDataAndActualData:-int(len(data_) * 0.1) , : ]
# dataTrain.reset_index(drop=True, inplace=True)
# dataTest = data_.iloc[-int(len(data_) * 0.1) : , : ]
# dataTest.reset_index(drop=True, inplace=True) 

################ data preprocessing  for forex STOCK ###########
preProcessedData = lobPreprocessing(NqPath, testNumbers)
trainData, testData, dataTrain, dataTest = preProcessedData.final()

###################################### loop fr train ###########

      ################## initialize ##################
       
modelSelection = modelForQ(trainData, outPutOfQ)
Q = modelSelection.fullyConnectedModel()
env = dict()
memory = deque(maxlen = relayMemorySize)
selectedAction = actions[random.randint(0, len(actions)-1)]

      #################################################

#len(trainData)-1


for i in range(100):

    ############################ current state #############
    state = trainData[i, :]
    
    ########################### transition between state and action #####################
    env_ = openOrClosePosition(env, selectedAction, i, dataTrain, margin,  Sl, Tp, levrage)   
    env = env_.tradeCheck() 

    ################################## next observation #################                  
    nextState = trainData[i+1, :] 
    
    ################################# reard calculation ######
    rewardCalculating = rewardCalculation(env, i, selectedAction, dataTrain, stepSwith = True)
    reward = rewardCalculating.reward()
    
    ################################### saving to memory ###################
    memory.append([state, actionConverting[actions.index(selectedAction)], reward, nextState])       
    
    
    ############################## action selection ####
    
    if len(memory) >= numberOfFirstBatchData:
        
        trainer = updatinQtable(memory, batchSize, Q, gamma, nextState)
        Q = trainer.updating()
        print("Q was updated")
        if epsilon > epsilonFinal:
          epsilon *= epsilonDecay
    
    ###########################################################  select action according to decided policy ##################

    policy_ = policy(Q, actions, state, epsilon)
    selectedAction = policy_.actionSelection()
    
    
    ############################# state updating ##########################
    state = nextState
    
    ############################## save Q ##########################


Q.save("DQN.csv")
    
    


############################## test section #########################

envTest = dict()
memoryTest = deque(maxlen = relayMemorySize)
selectedAction = actions[random.randint(0,len(actions)-1)]

len(testData)-1

for i in range(100):
    
    
    
    ############################ current state #############
    state = trainData[i, :]

    ########################### transition between state and action #####################
    env_ = openOrClosePosition(envTest, selectedAction, i, dataTest, margin,  Sl, Tp, levrage)   
    envTest = env_.tradeCheck() 

    ################################## next observation #################                  
    nextState = trainData[i+1, :] 
    
    ################################# reard calculation ######
    rewardCalculating = rewardCalculation(envTest, i, selectedAction, dataTest, stepSwith = True)
    reward = rewardCalculating.reward()
    
    ################################### saving to memory ###################
    memoryTest.append([state, actionConverting[actions.index(selectedAction)], reward, nextState])       
    
    
    ############################## action selection ####
    
    if len(memoryTest) >= numberOfFirstBatchData:
        
        trainer = updatinQtable(memoryTest, batchSize, Q, gamma, nextState)
        Q = trainer.updating()
        print("Q test was updated")
        if epsilonTest > epsilonFinal:
          epsilonTest *= epsilonDecay
    
    ###########################################################  select action according to decided policy ##################

    policy_ = policy(Q, actions, state, epsilonTest)
    selectedAction = policy_.actionSelection()
    
    
    ############################# state updating ##########################
    state = nextState


Q.save("DQNTest.csv")


################## show results #############################
showTraderResults_ = showTraderResults(envTest, dataTest["close"].values)
statements = showTraderResults_.final()







    
    
    