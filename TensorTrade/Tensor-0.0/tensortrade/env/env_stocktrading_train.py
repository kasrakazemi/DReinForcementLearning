################ Libs ##############
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stockstats import StockDataFrame 
import matplotlib.pyplot as plt
from zigzag import peak_valley_pivots
################################

"""
A custom stock trading environment based on OpenAI gym (Train Env)

"""

class StockTradingEnv_Train(gym.Env):

    def __init__(self,df, price_data, date, config,zigzag_data=None):

        self.data = df
        # self.zigzag_data = zigzag_data    # zigzag indicator that has been calculated on dataset
        #self.zigzag_pro = self._make_zigzag_pro()
        self.config = config
        self.reward_calculation = self.config["REWARD_CALCULATION"]     # determine reward function
        self.price_data = price_data
        self.price_data['action'] = 0   # make action column in price data to save each action
        self.atr = StockDataFrame.retype(price_data.copy())['atr'] # calculate atr indicator
        self.timesteps = self.config["TIMESTEPS"] 
        self.action_space = spaces.Discrete(3) # define action space
        self.state_space = (self.action_space.n * config["USE_LAST_ACTION"]) + 2 * config[ 
            "USE_CURRENT_PNL"] + self.timesteps * len(df.columns)  # define states space
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,)) # don't change this line !!!
        self.num_of_shares = config["MAX_NUM_SHARES"]
        #self.action_dimension = 1
        self.date = date
        #self.scaling_method = config["NORMALIZATION_METHOD"]
        #self.scaler = None
        daily = pd.to_datetime([pd.to_datetime(x) - pd.Timedelta(hours=15) for x in self.date]).floor('d') # specify day indexes
        daily = pd.DataFrame(daily, columns=['date'])
        self.day_indices = [x.total_seconds() for x in daily['date'] - daily['date'].shift(1)]
        self.day_indices = [i for i, v in enumerate(self.day_indices) if v != 0]
        self.day_indices = self.day_indices[1:] 
        self.day_indices[0] = self.timesteps 
        # if self.day_indices[1]-self.day_indices[0] < self.timesteps+100:
        #   self.day_indices.pop(0)
        # if len(self.data) -  self.day_indices[-1] < self.timesteps+100:   
        #    self.day_indices.pop(-1)
        self.current_point = self.day_indices[0] 
        self.day_index = 0 
        self.adjusted_reward = config["ADJUSTED_REWARD"]
        self.current_position = 0
        self.last_action = 0
        self.entry_index = 0
        #self.train_cols = config["TECHNICAL_INDICATORS_LIST"]
        self.spread_cost = config["Exchange_Commission"]    # set exchange commission
        #self.min_benefit = 0
        self.base_account = config["INITIAL_ACCOUNT_BALANCE"]
        self.reward_scaling = config["REWARD_SCALING"] # same as reward decay

        self.episode = 0
        self.account = self.base_account
        # memorize all the total balance change
        self.entry_date = None
        self.tradeslist = pd.DataFrame( 
            columns=['entry date', 'entry price', 'action', 'exit date', 'exit price', 'pnl','cum_pnl', 'mae', 'mpe'])
        self.tradeslist.loc[0] = [self.date[0],0, 0,
        self.date[0], 0,0, self.account, 0, 0 ]

        if self.reward_calculation =="SharpRatio_based":
            self.sharp_dataframe= pd.DataFrame(columns=['cum_pnl','cum_sum_pct'])
        self._seed()
        self.positives = 0    # ?
        self.negatives = 0    # ?
        self.step_counter=0

    '''
    func to determine stop loss base on zigzag indicator

    '''
    def find_pivots(self, action):

        pivot_param = .002
        step = .0005
        lookback = self.config['lookback']
        data = self.price_data['close'][max(0, self.current_point - lookback):self.current_point].to_numpy()
        pivots = peak_valley_pivots(data, pivot_param, -pivot_param)
        condition = -action
        current_price = self.price_data['close'][self.current_point - 1]
        if np.sum(pivots == condition) and condition * current_price < condition * data[
            np.where(pivots == condition)[0][-1]]:
            return abs(current_price - data[np.where(pivots == condition)[0][-1]])
        else:
            return self.config["STOP_LOSS_COEF"] * self.atr[self.current_point]     

    # def _make_zigzag_pro(self):
    #     zigzag_pro = pd.DataFrame([])
    #     data_len = len(self.zigzag_data)
    #     for col in self.zigzag_data.columns:
    #         pro = []
    #         next_index = next(iter(x for x in range(len(self.zigzag_data[col])) if self.zigzag_data[col][x] in [1, -1]),
    #                           data_len - 1)
    #         pivot_type = self.zigzag_data[col][next_index]
    #         for i, x in enumerate(self.zigzag_data[col].to_numpy()):
    #             if next_index <= i:
    #                 next_index = next(
    #                     iter(x + next_index + 1 for x in range(len(self.zigzag_data[col][next_index + 1:])) if
    #                          self.zigzag_data[col][x + next_index + 1] in [1, -1]), data_len - 1)
    #                 pivot_type = -pivot_type
    #             pro.append(pivot_type * (next_index - i))
    #         zigzag_pro[col] = pro.copy()
    #     return zigzag_pro

    '''
    func to get date

    '''
    def get_date(self):
        
        try:
            return self.date[self.current_point]
        except:
            pass

    '''
    func to get price

    '''
    def get_price(self, point):

        price_type = self.config["PRICE_CALCULATION_TYPE"]

        if price_type == 'mid':
            return self.price_data.loc[point, ['open', 'low', 'high', 'close']].mean()
        elif price_type == 'close':
            return self.price_data.loc[point,'close']
        elif price_type=='open':
            return self.price_data.iloc[point]['open']

    '''
    func to calculate PNL

    '''
    def calculate_pnl(self, action, previous_price, current_price, spread=True):

        if  action==0:  # if action be 0(hold) pnl will be 0 
            return 0
        diff = current_price - previous_price

        if spread:
            return (action * diff - self.spread_cost * spread ) * self.num_of_shares
        else:
            return (action * diff ) * self.num_of_shares
             

    '''
    func to calculate reward

    '''
    def calculate_reward(self, current_position, new_action, current_point, previous_point,sharp_data):

        previous_price = self.get_price(previous_point)
        current_price = self.get_price(current_point)
        next_price = self.get_price(current_point + 1)
        profit_loss = self.calculate_pnl(current_position, previous_price, current_price)
        profit_loss = (profit_loss if (current_position != new_action and new_action!=0) else 0)   # ignore profit_loss if new action be same as previous action
        min_price = self.price_data['low'][previous_point:current_point].min()
        max_price = self.price_data['high'][previous_point:current_point].max()

        if current_position == 1:
            MAE = self.calculate_pnl(current_position, previous_price, min_price, spread=False)
            MPE = self.calculate_pnl(current_position, max_price, current_price, spread=False)
        elif current_position == -1:
            MAE = self.calculate_pnl(current_position, previous_price, max_price, spread=False)
            MPE = self.calculate_pnl(current_position, min_price, current_price, spread=False)
        else:
            MAE = 0
            MPE = 0
        

        if self.reward_calculation == 'zigzag_based':
            col = self.zigzag_pro.columns[0]
            # reward = self.zigzag_pro[col][self.current_point] * new_action
            # if new_action in [1,-1] and reward > 0 : reward-= 10
            # # print(f'reward = {reward} for action = {new_action}')
            if current_position in [1, -1] and new_action != current_position:
                reward = profit_loss / 4
            else:
                act = -current_position if new_action == 0 else new_action
                col = self.zigzag_pro.columns[0]
                is_correct = self.zigzag_pro[col][self.current_point] * act
                reward = 0 if is_correct >= 0 else is_correct
                # for col in self.zigzag_data.columns:
                #     reward -= self.zigzag_data[col][self.current_point] * act
                # if act and not reward:
                if reward == 0 and act in [1, -1]:
                    pivot_data = self.zigzag_pro[col].to_list()
                    if -act in pivot_data[self.current_point:]:
                        next_pivot_index = pivot_data.index(-act, self.current_point)
                        distance_to_pivot = next_pivot_index - self.current_point
                        reward += distance_to_pivot / 10
                    else:
                        reward = is_correct

        elif self.reward_calculation == 'action_based':
            reward = self.calculate_pnl(new_action, current_price, next_price, spread=False)

        elif self.reward_calculation =="position_based" :
            reward = profit_loss

        elif self.reward_calculation =="SharpRatio_based" :
            sharpratio_data = sharp_data
            if len(sharpratio_data)< self.timesteps:
                reward = self.calculate_pnl(current_position, current_price, next_price, spread=False)
            else: 
                reward = ((sharpratio_data['cum_sum_pct'][-self.timesteps: ]).mean()+ 1e-9)/((sharpratio_data['cum_sum_pct'][-self.timesteps: ]).std()+ 1e-9)
              
        
        reward *= self.reward_scaling

        return profit_loss, reward, MAE, MPE

    '''
    func to calculate account PNL in percent

    '''
    def calculate_benefit(self):

        action = self.current_position
        profit_loss = self.calculate_pnl(action, self.get_price(self.entry_index),
                                         self.get_price(self.current_point))
        acount_pnl_percentage = (self.base_account + profit_loss) / self.base_account

        return (acount_pnl_percentage - 1), profit_loss

    '''
    func to get environment state

    '''
    def get_state(self):

        s_ = self.data.iloc[self.current_point - self.timesteps:self.current_point , :]
        if self.config["WINDOW_NORMALIZE"]:
            for column_name in ['close', 'low', 'high', 'open']:
                if column_name in s_.columns:
                    window_data = self.price_data.iloc[self.current_point - self.timesteps:self.current_point ,
                                  column_name]
                    min_val = np.min(window_data)
                    max_val = np.max(window_data)
                    s_[column_name] = (window_data - min_val) / (max_val - min_val)
        s_ = s_.values.reshape(-1)
        # print(self.current_point, self.timesteps, s_.shape)
        extended_data = np.zeros(
            self.action_space.n * self.config["USE_LAST_ACTION"] + 2 * self.config["USE_CURRENT_PNL"])
        if self.config["USE_LAST_ACTION"]:
            extended_data[self.current_position + 1] = 1
        if self.config["USE_CURRENT_PNL"]:
            benefit, pnl = self.calculate_benefit()
            # if benefit:
            # s   benefit = np.sign(benefit)
            extended_data[-1] = benefit
            extended_data[-2] = ((pnl + self.account) / self.base_account - 1) * 10
        st = np.expand_dims(np.append(s_, extended_data), 0)
       
        return st

    '''
    func to step action on environment 

    '''
    def step(self, action):
        
        reward = 0
        action = int(action) - 1    # action space should be in (-1,0,1)
        #self.price_data['action'][self.current_point] = action
        self.price_data.loc[self.current_point,'action']= action
        
        done = 0
        # self.current_price = self.next_price
        # self.next_price = self.get_price(self.current_point + 1)
        if (self.day_index < len(self.day_indices) and self.current_point == self.day_indices[self.day_index] - 1) or \
                self.current_point == len(self.data) - 2:
            done = 1 
            # action = 0
            
        # if action == self.current_position and self.current_position:
        #     previous_price = self.get_price(self.entry_index)
        #     current_price = self.get_price(self.current_point)
        #     diff = current_price - previous_price
        #     temp = (action * diff - self.spread_cost) * self.num_of_shares
        #     # Check stop loss
        #     sl = self.find_pivots(action)
        #     if temp < -sl:
        #         action = 0
        profit_loss, reward, MAE, MPE = self.calculate_reward(self.last_action, action, self.current_point,self.entry_index,self.sharp_dataframe)
        
        if self.step_counter==0:                                                    
                self.sharp_dataframe.loc[len(self.sharp_dataframe)]= [profit_loss+self.base_account,0]
        else:
                self.sharp_dataframe.loc[len(self.sharp_dataframe)]= [(profit_loss+self.sharp_dataframe.loc[len(self.sharp_dataframe)-1]['cum_pnl'])
                                                                    ,((profit_loss+self.sharp_dataframe.loc[len(self.sharp_dataframe)-1]['cum_pnl'])/self.sharp_dataframe.loc[len(self.sharp_dataframe)-1]['cum_pnl'])*1 -1]
        if action!=0 and self.last_action!= action :
          
                #print('action:',doned_action,'position:',self.current_position)
                self.tradeslist.loc[len(self.tradeslist)] = [self.entry_date, self.get_price(self.entry_index),
                                                            self.last_action, self.get_date(),
                                                            self.get_price(self.current_point), profit_loss,self.account, MAE,
                                                            MPE]
                self.entry_index = self.current_point
                self.entry_date = self.get_date()

                self.last_action= action
            
        if profit_loss:
            if profit_loss > 0:
                self.positives += profit_loss
            else:
                self.negatives += profit_loss
            self.account += profit_loss
           
        # if  action != last_action:
        #         self.entry_index = self.current_point
        #         self.entry_date = self.get_date()

        self.current_position = action
        if self.account <= 0:
            done = 1

        self.current_point += 1
        self.step_counter+=1
        s_ = self.get_state()
      
        return s_, float(reward), done, {}

    '''
    func to reset environment

    '''
    def reset(self):

        self.positives = 0
        self.negatives = 0
        self.account = self.base_account
        self.current_point = self.day_indices[self.day_index]
        self.day_index = (self.day_index + 1) % (len(self.day_indices))

        #self.make_plot()      # plot result at the end of each day(episode)
        self.episode += 1
        #print(self.episode)
        self.current_position = 0
        self.entry_index = self.current_point

        return self.get_state()

    '''
    func to ؟

    '''
    def _seed(self, seed=None):     # we want to start with same arameters each time
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    '''
    func to ؟

    '''
    def get_sb_env(self):      # ??
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    '''
    func to plot result

    '''
    def make_plot(self):
        plt.plot(self.tradeslist['pnl'], 'g')   # plot PNL from trade list
        plt.savefig('results/PNL.png')      # save plot as png
        plt.close()


