########### Libs ##########
import json
###########################

Config_Path= "E:/Tensurf-RL/tensurfrl/TensorTrade/Tensor-0.1/configuration.json"
Config_File= open(Config_Path)
Config = json.load(Config_File)

Data_Path = Config['Data_Path']
Run_mode= Config['RUNNING_MODE']
Feature_engineer= Config['Feature_engineering']
Train_env_config = Config['env_config_train']
Test_env_config = Config['env_config_test']
Model_params = Config['A2C_PARAMS']
Model_name= Config['MODEL_NAME']


