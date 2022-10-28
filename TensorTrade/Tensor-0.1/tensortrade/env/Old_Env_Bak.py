#####
import inspect
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stockstats import StockDataFrame as Sdf
from zigzag import peak_valley_pivots


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df, price_data, date, timesteps, config,
                 zigzag_data=None, is_for_train=False):
        self.data = df
        self.zigzag_data = zigzag_data
        self.zigzag_pro = self._make_zigzag_pro()
        self.config = config
        self.reward_calculation = self.config["REWARD_CALCULATION"]
        self.price_data = price_data
        self.price_data['action'] = 0
        self.atr = Sdf.retype(price_data.copy())['atr'].reset_index(drop=True)
        self.atr.index = price_data.index
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.timesteps = timesteps
        self.action_space = spaces.Discrete(3)
        self.state_space = (self.action_space.n * config["USE_LAST_ACTION"]) + 2 * config[
            "USE_CURRENT_PNL"] + self.timesteps * len(df.columns)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.is_for_train = is_for_train
        self.num_of_shares = config["MAX_NUM_SHARES"]
        self.action_dimension = 1
        self.date = date
        self.scaling_method = config["NORMALIZATION_METHOD"]
        self.scaler = None
        daily = pd.to_datetime([pd.to_datetime(x) - pd.Timedelta(hours=15) for x in self.date]).floor('d')
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
        self.current_position = 2
        self.entry_index = 0
        self.train_cols = config["TECHNICAL_INDICATORS_LIST"]
        self.spread_cost = config["SPREAD_COST"]
        self.min_benefit = 0
        self.base_account = config["INITIAL_ACCOUNT_BALANCE"]
        self.reward_scaling = config["REWARD_SCALING"]
        self.model_name = config["MODEL_NAME"]
        # self.previous_price = 0
        # self.current_price = 0
        # self.next_price = 0
        self.episode = 0
        self.account = self.base_account
        # memorize all the total balance change
        self.entry_date = None
        self.tradeslist = pd.DataFrame(
            columns=['entry date', 'entry price', 'action', 'exit date', 'exit price', 'pnl', 'mae', 'mpe'])
        self._seed()
        self.positives = 0
        self.negatives = 0

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

    def _make_zigzag_pro(self):
        zigzag_pro = pd.DataFrame([])
        data_len = len(self.zigzag_data)
        for col in self.zigzag_data.columns:
            pro = []
            next_index = next(iter(x for x in range(len(self.zigzag_data[col])) if self.zigzag_data[col][x] in [1, -1]),
                              data_len - 1)
            pivot_type = self.zigzag_data[col][next_index]
            for i, x in enumerate(self.zigzag_data[col].to_numpy()):
                if next_index <= i:
                    next_index = next(
                        iter(x + next_index + 1 for x in range(len(self.zigzag_data[col][next_index + 1:])) if
                             self.zigzag_data[col][x + next_index + 1] in [1, -1]), data_len - 1)
                    pivot_type = -pivot_type
                pro.append(pivot_type * (next_index - i))
            zigzag_pro[col] = pro.copy()
        return zigzag_pro

    def _get_date(self):
        try:
            return self.date[self.current_point]
        except:
            print(self.data)
            print('current_point=', self.current_point)
            return self.date[self.current_point]

    def reset(self):
        # print(inspect.stack()[1].function)
        #        print('============',self.date[self.day_indices[self.day_index]])
        self.positives = 0
        self.negatives = 0
        if self.is_for_train:
            self.account = self.base_account
            self.current_point = self.day_indices[self.day_index]
            self.day_index = (self.day_index + 1) % (len(self.day_indices))
        else:
            self.current_point = self.day_indices[self.day_index]
            self.day_index = self.day_index + 1
        #           print (self.current_point)
        # print(self._get_date())
        # self.current_price = self.get_price(self.current_point)
        # self.previous_price = self.current_price
        # self.next_price = self.get_price(self.current_point+1)
        self.episode += 1
        self.current_position = 0
        self.entry_index = self.current_point
        return self.get_state()

    def render(self, mode='human', close=False):
        return self.get_state()

    ## TODO: open of next candle
    def get_price(self, point, source=False):
        # return self.price_data.loc[point,'close']
        if source:
            return self.price_data.loc[point, 'open']
        return self.price_data.loc[point, 'open']
        # return self.price_data.loc[point, ['open', 'low', 'high', 'close']].mean()
        # type = self.config["PRICE_CALCULATION_TYPE
        # if type == 'mid':
        #     return self.price_data.loc[point, ['open', 'low', 'high', 'close']].mean()
        # elif type == 'close':
        #     return self.price_data.loc[point,'close']
        # else: #  Random price
        #     a = self.price_data.loc[point,'high']
        #     b = self.price_data.loc[point,'low']
        #     price = np.random.rand() * (a-b) + b
        #     return price

    def calculate_pnl(self, action, previous_price, current_price, spread=True):
        if not action:
            return 0
        diff = current_price - previous_price
        profit_loss = (action * diff - self.spread_cost * spread if action else 0) * self.num_of_shares
        return profit_loss

    def calculate_reward(self, done_action, new_action, current_point, previous_point):
        previous_price = self.get_price(previous_point, True)
        current_price = self.get_price(current_point, True)
        next_price = self.get_price(current_point + 1, True)
        profit_loss = self.calculate_pnl(done_action, previous_price, current_price)
        profit_loss = profit_loss if done_action != new_action else 0
        min_price = self.price_data['low'][previous_point:current_point].min()
        max_price = self.price_data['high'][previous_point:current_point].max()

        if done_action == 1:
            MAE = self.calculate_pnl(done_action, previous_price, min_price, spread=False)
            MPE = self.calculate_pnl(done_action, max_price, current_price, spread=False)
        elif done_action == -1:
            MAE = self.calculate_pnl(done_action, previous_price, max_price, spread=False)
            MPE = self.calculate_pnl(done_action, min_price, current_price, spread=False)
        else:
            MAE = 0
            MPE = 0
        if self.reward_calculation == 'zigzag_based':
            col = self.zigzag_pro.columns[0]
            # reward = self.zigzag_pro[col][self.current_point] * new_action
            # if new_action in [1,-1] and reward > 0 : reward-= 10
            # # print(f'reward = {reward} for action = {new_action}')
            if done_action in [1, -1] and new_action != done_action:
                reward = profit_loss / 4
            else:
                act = -done_action if new_action == 0 else new_action
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
            next_profit_loss = self.calculate_pnl(new_action, current_price, next_price, spread=False)
        else:  # "position_based"
            reward = profit_loss

        reward *= self.reward_scaling
        return profit_loss, reward, MAE, MPE

    def _make_plot(self):
        plt.plot(self.tradeslist['pnl'], 'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def calculate_benefit(self):
        action = self.current_position
        profit_loss = self.calculate_pnl(action, self.get_price(self.entry_index, True),
                                         self.get_price(self.current_point, True))
        acc_percent = (self.base_account + profit_loss) / self.base_account
        return (acc_percent - 1), profit_loss

    def get_state(self):
        s_ = self.data.loc[self.current_point - self.timesteps:self.current_point - 1, :]
        if self.config["WINDOW_NORMALIZE"]:
            for column_name in ['close', 'low', 'high', 'open']:
                if column_name in s_.columns:
                    window_data = self.price_data.loc[self.current_point - self.timesteps:self.current_point - 1,
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

    def step(self, action):
        reward = 0
        action = int(action) - 1
        self.price_data['action'][self.current_point] = action
        # with open('data4metalabeling.csv','at') as ff:
        #   ff.write(f'{self.date[self.current_point-1]},{int(action)}\n')
        # with open('file.txt','at') as f:
        #    f.write(f'{self.date[self.current_point]},{self.price_data["close"][self.current_point]},{action}\n')
        done = 0
        # self.current_price = self.next_price
        # self.next_price = self.get_price(self.current_point + 1)
        if self.day_index < len(self.day_indices) and self.current_point == self.day_indices[self.day_index] - 1 or \
                self.current_point == len(self.data) - 2:
            done = 1
            action = 0
        if action == self.current_position and self.current_position:
            previous_price = self.get_price(self.entry_index, True)
            current_price = self.get_price(self.current_point, True)
            diff = current_price - previous_price
            temp = (action * diff - self.spread_cost) * self.num_of_shares
            # Check stop loss
            sl = self.find_pivots(action)
            if temp < -sl:
                action = 0
        profit_loss, reward, MAE, MPE = self.calculate_reward(self.current_position, action, self.current_point,
                                                              self.entry_index)
        # if reward:
        #    print(reward)
        if profit_loss:
            if profit_loss > 0:
                self.positives += profit_loss
            else:
                self.negatives += profit_loss
            self.account += profit_loss
            self.tradeslist.loc[len(self.tradeslist)] = [self.entry_date, self.get_price(self.entry_index, True),
                                                         self.current_position, self._get_date(),
                                                         self.get_price(self.current_point, True), profit_loss, MAE,
                                                         MPE]
        if not self.current_position or action != self.current_position:
            self.entry_index = self.current_point
            self.entry_date = self._get_date()
            # self.previous_price = self.current_price
        self.current_position = action
        if self.account <= 0:
            done = 1
        self.current_point += 1
        s_ = self.get_state()
        if not self.is_for_train and (
                self.current_point >= len(self.data) - 2 or self.day_index == len(self.day_indices) - 1):
            return s_, float(reward), done, 'finish'
        return s_, float(reward), done, {}

    def save_asset_memory(self):
        if not self.is_for_train:
            self.price_data.to_csv('action_evaluation.csv')
        return self.tradeslist

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

