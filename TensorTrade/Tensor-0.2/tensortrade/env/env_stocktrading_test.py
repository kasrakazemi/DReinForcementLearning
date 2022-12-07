################ Libs ##############
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stockstats import StockDataFrame 
from zigzag import peak_valley_pivots
from tabulate import tabulate
from scipy.signal import argrelextrema
################################

"""
A custom stock trading environment based on OpenAI gym (Test Env)

"""

class StockTradingEnv_Test(gym.Env):

    def __init__(self,df, price_data, date, config):
       
        self.data = df
        self.config = config
        self.price_data = price_data
        self.price_data['action'] = 0   # make action column in price data to save each action
        self.atr = StockDataFrame.retype(price_data.copy())['atr'] # calculate atr indicator
        self.timesteps = self.config["TIMESTEPS"] 
        self.run_mode= config['Run_Mode']
        self.action_space = spaces.Discrete(3) # define action space
        self.state_space = (self.action_space.n * config["USE_LAST_ACTION"]) + 2 * config[ 
            "USE_CURRENT_PNL"] + self.timesteps * len(df.columns)  # define states space
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,config['TIMESTEPS'],49)) # don't change this line !!!
        self.num_of_shares = config["MAX_NUM_SHARES"]
        self.date = date # date of dataframe

        daily = pd.to_datetime([pd.to_datetime(x) - pd.Timedelta(hours=15) for x in self.date]).floor('d') # specify day indexes
        daily = pd.DataFrame(daily, columns=['date'])
        self.day_indices = [x.total_seconds() for x in daily['date'] - daily['date'].shift(1)]
        self.day_indices = [i for i, v in enumerate(self.day_indices) if v != 0]
        self.day_indices = self.day_indices[1:] 
        self.day_indices[0] = self.timesteps 

        self.current_point = self.day_indices[0] 
        self.day_index = 0 
        self.current_position = 0
        self.last_action = 0
        self.entry_index = 0
       
        self.spread_cost = config["Exchange_Commission"]    # set exchange commission
        
        self.base_account = config["INITIAL_ACCOUNT_BALANCE"]

        self.episode = 0
        self.account = self.base_account
        # memorize all the total balance change
        self.entry_date = None
        self.tradeslist = pd.DataFrame( 
            columns=['entry date', 'entry price', 'action', 'exit date', 'exit price', 'pnl','cum_pnl', 'mae', 'mpe','exit mode','sl','tp'])
        self.tradeslist.loc[0] = [self.date[0],0, 0,
        self.date[0], 0,0, self.account, 0, 0, 0, 0, 0]
       
        self.positives = 0    # ?
        self.negatives = 0    # ?

        self.sl_mode= config["SL_mode"]
        self.sl_value= config["STOP_LOSS"]                                
        self.scipy_neighborhood=  config["Scipy_neighborhood"]
        self.R2R = config["R2R"]
        self.stop_loss= 0
        self.take_profit = -1
        self.sltp_flag= False

        
        
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
            return self.price_data.iloc[point, ['open', 'low', 'high', 'close']].mean()
        elif price_type == 'close':
            return self.price_data.iloc[point,'close']
        elif price_type=='open':
            return self.price_data.iloc[point]['open']

    '''
    func to calculate PNL

    '''
    def calculate_pnl(self, action, previous_price, current_price, spread=True):

        if  action==0:  # if action be 0(hold) pnl will be 0 
            return 0

        diff_price = current_price - previous_price

        if spread:
            return (action * diff_price - self.spread_cost * spread ) * self.num_of_shares
        else:
            return (action * diff_price ) * self.num_of_shares
             
    '''
    func to calculate reward

    '''
    def calculate_reward(self,current_position, new_action, current_point, previous_point):

        previous_price = self.get_price(previous_point)
        current_price = self.get_price(current_point)
        next_price = self.get_price(current_point + 1)
        profit_loss = self.calculate_pnl(current_position, previous_price, current_price)
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

        return profit_loss, MAE, MPE

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
        st= s_.values.reshape(1,100,49)
       
        return st

    '''
    func calculate sl tp based on swing

    '''
    def calculate_sltp(self, current_index, action, fill_price):
        
        minimum_risk = self.sl_value
        self.risk = self.sl_value
        if self.sl_mode == 'fixed':  #fix mode
            if action == 1:
                self.stop_loss = fill_price - action * self.risk
                self.take_profit = fill_price + action * self.risk 
            else:
                 self.take_profit= fill_price + action * self.risk
                 self.stop_loss= fill_price - action * self.risk 

        else:   #Swing mode
            if action == 1:
                low_data = self.price_data['low'][:current_index + 1]
                swing_low_indices = argrelextrema(np.array(low_data), lambda x, y: x < y, order=self.scipy_neighborhood, )[0]
                try:
                    self.stop_loss = low_data[next(x for x in swing_low_indices[::-1] if low_data[x] < fill_price)]
                except: 
                    self.stop_loss = fill_price - minimum_risk
                self.risk = abs(self.stop_loss - fill_price)
                self.take_profit = fill_price + self.risk * self.R2R

            else:
                high_data = self.price_data['high'][:current_index+1]
                swing_high_indices = argrelextrema(np.array(high_data), lambda x, y: x > y, order=self.scipy_neighborhood)[0]
                try:
                    self.stop_loss =  high_data[next(x for x in swing_high_indices[::-1] if high_data[x] > fill_price)]
                except:
                    self.stop_loss = fill_price + minimum_risk
                self.risk = abs(self.stop_loss - fill_price)
                self.take_profit = fill_price - self.risk * self.R2R

    '''
    func to step action on environment 

    '''
    def step(self, action):
        
        doned_action = int(action) - 1    # action space should be in (-1,0,1)
        reward=0
        self.price_data.loc[self.current_point,'action']= doned_action
        # last_action = (self.tradeslist)['action'][len(self.tradeslist)-1]
        done = 0
        close_price = self.get_price(self.current_point)

        if (self.day_index < len(self.day_indices)-1 and self.current_point == self.day_indices[self.day_index+1] -1 ) or \
                self.current_point == len(self.data) - 2:
            done = 1 

        if  (doned_action!=0 and self.sltp_flag):

            self.entry_index = self.current_point
            self.entry_date = self.get_date()
            self.calculate_sltp(self.current_point,doned_action,close_price)
            self.last_action= doned_action
            self.sltp_flag= False

        if (doned_action!=0 and self.last_action!= doned_action) :

            profit_loss, MAE, MPE = self.calculate_reward(self.last_action,doned_action, self.current_point,self.entry_index)
            if profit_loss > 0:
                self.positives += profit_loss
            else:
                self.negatives += profit_loss
            self.account += profit_loss
            #print('sl: ',self.stop_loss,'tp: ',self.take_profit,'action: ',self.last_action)
            self.tradeslist.loc[len(self.tradeslist)] = [self.entry_date, self.get_price(self.entry_index),
                                                        self.last_action, self.get_date(),
                                                        self.get_price(self.current_point), profit_loss,self.account, MAE,
                                                        MPE,'reverse',self.stop_loss,self.take_profit]
            self.entry_index = self.current_point
            self.entry_date = self.get_date()
            self.calculate_sltp(self.current_point,doned_action,close_price)
            self.last_action= doned_action
            #self.sltp_flag= False
            
        elif (self.last_action== 1  and (close_price>= self.take_profit or close_price<= self.stop_loss)) or (self.last_action== -1 and (close_price<= self.take_profit or close_price>= self.stop_loss)):

                profit_loss, MAE, MPE = self.calculate_reward(self.last_action,doned_action, self.current_point,self.entry_index)
                if profit_loss > 0:
                 self.positives += profit_loss
                else:
                    self.negatives += profit_loss
                self.account += profit_loss
                #print('action:',doned_action,'close:',close_price,'sl:',self.stop_loss,'tp:',self.take_profit)
                self.tradeslist.loc[len(self.tradeslist)] = [self.entry_date, self.get_price(self.entry_index),
                                                            self.last_action, self.get_date(),
                                                            self.get_price(self.current_point), profit_loss,self.account, MAE,
                                                            MPE,'sltp',self.stop_loss,self.take_profit]

                self.last_action= 0
                self.sltp_flag= True
        
        self.current_position = doned_action
        
        if self.account <= 0:
            done = 1
            
        self.current_point += 1
       
        s_ = self.get_state()
      
        return s_, float(reward), done, {"account_status":self.account}

    '''
    func to reset environment

    '''
    def reset(self):

        self.positives = 0
        self.negatives = 0
        self.day_index +=1
        self.current_point = self.day_indices[self.day_index]
    
        #self.make_plot() # plot result at the end of each day(episode)
        self.episode += 1
       
        self.current_position = 0
        self.entry_index = self.current_point

        return self.get_state()
        
    '''
    func to ?

    '''
    def get_sb_env(self):     
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
    
    '''
    func to save actions that has been taken on each step

    '''
    def trade_history(self):

        self.price_data.to_csv('actiones_taked.csv')

        return self.tradeslist


    '''
    func to get results of backtesting

    '''
    def show_results(self):

        results= self.tradeslist.copy()
        #results['cum_pnl'] = results['pnl'].cumsum()
        initial_balance= self.base_account
        cagr_dd = -20
        pf = 0
        pessimistic_return_on_margin = -10
        
        if len(results):
            pos = np.sum(results['pnl'][results['pnl'] > 0])
            neg = np.sum(results['pnl'][results['pnl'] < 0])
            long_num = len(results['action'][results['action'] == 1])
            short_num = len(results['action'][results['action'] == -1])
            cagr = float(results['cum_pnl'][-1:])
            max_drawdown = np.min(
                results['cum_pnl'] - results['cum_pnl'].rolling(len(results), min_periods=1).max())
            mean_drawdown = -np.mean(
                results['cum_pnl'] - results['cum_pnl'].rolling(len(results), min_periods=1).max())
            cagr_dd = cagr / mean_drawdown if mean_drawdown != 0 else cagr
            pf = pos / -neg if neg else np.sum(results['pnl'] > 0)
            margin = 50
            average_win = 0
            average_loss = 0
            winning_trades = len(results[results['pnl'] > 0].index)
            losing_trades = len(results[results['pnl'] < 0].index)
            if winning_trades:
                average_win = pos / winning_trades
            if losing_trades:
                average_loss = - neg / losing_trades
            adjusted_num_of_wins = winning_trades - winning_trades ** .5
            adjusted_num_of_losses = losing_trades + losing_trades ** .5
            adjusted_profit = average_win * adjusted_num_of_wins
            adjusted_loss = average_loss * adjusted_num_of_losses
            pessimistic_return_on_margin = (adjusted_profit - adjusted_loss) / margin

            if self.run_mode in ('backtest','train'):
                print(tabulate(
                    {"long#":[long_num], "short#":[short_num], "Cumulative PnL": [cagr], "PROM": [pessimistic_return_on_margin],
                    "Profit factor": [pf], "max drawdown": [max_drawdown], "PnLs": {str(pos) + ' ' + str(neg)}}, headers="keys", tablefmt="grid"))
                    
                plt.figure(figsize=(9,6))
                plt.plot(results['exit date'],results['cum_pnl'] , color ='green', markersize=4)
                plt.grid()
                plt.ylabel('Point', labelpad = 8, fontsize = 14)
                plt.xlabel('Dates', labelpad = 8, fontsize = 14)
                plt.xticks(rotation=90)
                plt.savefig('results/PNL.png') 
                plt.show()

            else:
                return (pessimistic_return_on_margin,pf)
                
