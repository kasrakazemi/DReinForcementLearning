########### Libs ##########
import json
###########################

Config_Path= "E:/Tensurf-RL/tensurfrl/TensorTrade/configuration.json"
Config_File= open(Config_Path)
Config = json.load(Config_File)

Data_Path = Config['Data_Path']
Train_Percentage= Config['TRAIN_DATA_PERCENTAGE']
Test_Percentage= Config['TEST_DATA_PERCENTAGE']
window_size= Config['feature_extraction_params'][0]
lob_percent = Config['feature_extraction_params'][1]
Normalization_mode= Config['NORMALIZATION_METHOD']
Model_name= Config['MODEL_NAME']