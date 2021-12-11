# common library
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from datetime import timedelta  


from stable_baselines.common.vec_env import DummyVecEnv
import gym
from gym.utils import seeding
from gym import spaces

# RL models from stable-baselines
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import DDPG

from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# TO SAVE DEBUGGING FILE (will increase proocessing time)
SAVE_DEBUG_FILES = False
# Working data with technical indicators
WORKING_DATA_WITH_TE_PATH = "data/wd_te.csv"
# We will retrain our models after 60 business days
RETRAIN_MODEL_CYCLE = 60
# Validation window
VALIDATION_WINDOW = 60
# Investable assets 
INVESTABLE_ASSETS = dict(# Bonds
                          bonds = ["LQD", "SHY", "IEF", "TLT", "AGG"], 
                          # Equities  
                          equities = ["IJH", "IJR", "IVV", "IVE", "IVW","SPY"], # "^GSPC"
                          # Commodities
                          commodities = ["GLD"])

# TRANSACTION FEE
TRANSACTION_FEE_PERCENT = 0.0001 #1bps
# Gamma or reward scaling 
REWARD_SCALING = 1e-4

# Set deprecated logging off.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#------------------------------------------------------------------------------------------------
class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df, params):
        # TBD: Check if this env is created for each day of training.
        # My guess is that it is created once for each simulation history
        # For intermediate calls only the step is called
        self._seed(144)
        self.params = params

        # If training mode, we don't need to return lots of information
        # else we can return the holdings and other information from the
        # function
        self.train_mode = True
        if "train_mode" in self.params:
            self.train_mode = self.params["train_mode"]

        # make these members to avoid recalculations later in step
        self.num_asset_classes = len(self.params["investable_assets"].keys())
        self.bonds = self.params["investable_assets"]["bonds"]
        self.equities = self.params["investable_assets"]["equities"]
        self.commodities = self.params["investable_assets"]["commodities"]
        self.investable_tic = self.bonds + self.equities + self.commodities

        self.df = df.copy()
        self.df_unique_dates = self.df.Date.unique()
        self.non_investable_tic = list(set(self.df.tic) - set(self.investable_tic))
        
        # We keep ordered once for all
        seq_df = pd.DataFrame([(x[1],x[0]) for x in enumerate(self.investable_tic + self.non_investable_tic)]).rename(columns = {0:"tic", 1:"seq"})
        self.df = self.df.merge(seq_df, 
                      how="left", 
                      left_on = ["tic"], 
                      right_on = ["tic"])\
                     .sort_values(by=["Date","seq"])

        self._common_init_reset()
        pass

    def reset(self):
        self._common_init_reset()
        return self.state

    def _state_creator(self, p_data_for_day, p_asset_weights_dict, p_asset_weights=None):
        """
        p_data_for_day: dataframe of day data from which state will be constructed
        
        Pass EITHER 
        p_asset_weights_dict: dict with keys self.investable_tic
        OR
        p_asset_weights: asset weights in self.investable_tic order. if passed it is used. Else it is constructed from p_asset_weights_dict
        """
        
        asset_weights = None
        if p_asset_weights is None:
            assert len(p_asset_weights_dict) == len(self.investable_tic), "p_asset_weights_dict has not got needed assets"
            # This is in order -> all bonds wt + all equity wts + all commodity wts
            asset_weights = [p_asset_weights_dict[tic] for tic in self.investable_tic]
        else:
            asset_weights = p_asset_weights

        tmp_state = asset_weights +\
                    p_data_for_day.macd.values.tolist() + \
                    p_data_for_day.rsi.values.tolist() + \
                    p_data_for_day.cci.values.tolist() + \
                    p_data_for_day.adx.values.tolist() + \
                    [p_data_for_day.turbulence.values[0]]
        return tmp_state


    def _common_init_reset(self):
        self.day = 0
        # These are must for any environment: terminal, state, reward, action_space,observation_space
        self.terminal = False        

        self.data = self.df.loc[self.df.Date == self.df_unique_dates[0]]

        # state:
        init_asset_weights = None
        if "custom_init_weight" in self.params:
            init_asset_weights = self.params["custom_init_weight"]
        else:
            # default
            bond_weight = 0.35
            equity_weight = 0.60
            commodity_weight = 0.05
            init_asset_weights = {tic:wt for tic,wt in  zip(self.bonds + self.equities + self.commodities,
                                        [bond_weight * 1.0/len(self.bonds) for _ in self.bonds] +\
                                        [equity_weight * 1.0/len(self.equities) for _ in self.equities] +\
                                        [commodity_weight * 1.0/len(self.commodities) for _ in self.commodities])}

        self.state =  self._state_creator(p_data_for_day = self.data, p_asset_weights_dict = init_asset_weights)

        # Store once to quickly access later in step
        self.state_offset_dict = dict( 
                                       asset_distr_weights = (0, len(self.investable_tic))                    
                                     )

        # initialize reward
        self.reward = 0
        self.cost = 0
        # action_space: original paper takes -1 to 1 which means sell 100% or buy 100%
        # here we take 3 level of actions. 
        # First: set 3 elements are proportional weights of bond, equity and commodity
        # Second: next bonds_distr_wt + equity_distr_wt + commodity_distr_wt
        self.action_space = spaces.Box(low = 0.00000001, high = 1, shape = (len(self.investable_tic),))
        # observation_space: same as self.state - should correspond one to one  
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (len(self.state),))
        # memorize all the total balance change
        self.asset_memory = [self.params['initial_account_balance']]
        self.tc_cost_memory = [np.NaN]
        self.rewards_memory = []
        # self.weights_memory = pd.DataFrame(columns=["Date"] + self.investable_tic)


    def step(self, actions):
        
        # Terminal state is market if it is second last day of training set
        self.terminal = (self.day == len(self.df_unique_dates)-2)

        if np.isnan(actions).any():
                print("DEBUG: check why action is nan")
        #if self.day == 51:
        #    print("DEBUG: check why action becomes nan later")
        
        # beginning of period assets 
        investable_prev_weights = self.state[self.state_offset_dict['asset_distr_weights'][0]:self.state_offset_dict['asset_distr_weights'][1]]
        investable_prev_weights = np.array(investable_prev_weights)/sum(investable_prev_weights)
        begin_total_asset = self.asset_memory[self.day]

        self.day += 1
        self.data = self.df.loc[self.df.Date == self.df_unique_dates[self.day]]

        investable_returns = self.data.returns_close.values[0:len(self.investable_tic)] # note that it is already ordered in tic seq
        pre_rebal_pos = (investable_prev_weights * begin_total_asset) * (1.0 + investable_returns)
        end_total_asset = sum(pre_rebal_pos)

        investable_new_weights = actions[0:len(self.investable_tic)]
        investable_new_weights = investable_new_weights/investable_new_weights.sum()
        new_desired_rebal_pos  = investable_new_weights * end_total_asset

        transaction_fee = sum(np.abs(pre_rebal_pos - new_desired_rebal_pos)*TRANSACTION_FEE_PERCENT)
        self.reward = ((end_total_asset - transaction_fee)/begin_total_asset) - 1 
        # self.rewards_memory.append(self.reward)
        self.asset_memory.append(end_total_asset)
        self.tc_cost_memory.append(transaction_fee/begin_total_asset)

        # We also need to update new state
        self.state =  self._state_creator(p_data_for_day = self.data, p_asset_weights_dict = None, p_asset_weights = investable_new_weights.tolist())
        self.reward = self.reward * REWARD_SCALING

        info = {}
        if not self.train_mode:
            # Add weights for each asset: new_weight (post rebal weight), pre_weights = (pre rebal weight)
            info["weights"] = self._pre_post_rebal_weights(self.data.Date.values[0], self.investable_tic, investable_new_weights, pre_rebal_pos)
            info["value_returns_df"] = None

        if self.terminal and (self.train_mode == False):
            value_returns_df = pd.DataFrame({"Date": self.df_unique_dates, "account_value":self.asset_memory})
            value_returns_df['daily_return']=value_returns_df["account_value"].pct_change(1)
            value_returns_df["tc_cost"] = self.tc_cost_memory
            info["value_returns_df"] = value_returns_df

        return self.state, self.reward, self.terminal,info

    
    def _pre_post_rebal_weights(self, dt, tics,  investable_new_weights, pre_rebal_pos):
        tmp_weights = pd.DataFrame([(dt, x[0],x[1]) for x in zip(tics, investable_new_weights)])\
                                    .rename(columns = {0:"Date", 1: "tic", 2:"new_weight"})
        tmp_weights["pre_weights"] = pre_rebal_pos/sum(pre_rebal_pos)
        return tmp_weights


    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



#------------------------------------------------------------------------------------------------
def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    # model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', np.round((end - start) / 60, 2), ' minutes')
    return model

#------------------------------------------------------------------------------------------------
def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    #model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', np.round((end - start) / 60, 2), ' minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    # model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', np.round((end - start) / 60, 2),' minutes')
    return model


#------------------------------------------------------------------------------------------------
def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    #for i in range(len(test_data.index.unique())):
    #    action, _states = model.predict(test_obs)
    #    test_obs, rewards, dones, info = test_env.step(action)

    # We run ne date less to test till last. The second last date is the termination date
    dates = test_data.Date.unique()
    rewards, done, info = None, None, None
    weights_list = []
    value_returns_df = None
    for dt in dates[0:(len(dates)-1)]:
        action, _states = model.predict(test_obs)
        test_obs, rewards, done, info = test_env.step(action)
        if info[0]['weights'] is not None:
            weights_list.append(info[0]['weights'])
        if done[0]:
            value_returns_df = info[0]['value_returns_df']

    weights_df = pd.concat(weights_list)
    return value_returns_df, weights_df

#------------------------------------------------------------------------------------------------
def get_sharpe(value_returns_df):
    ###Calculate Sharpe ratio based on validation results###
    sharpe = (np.sqrt(252.0)) * value_returns_df['daily_return'].mean() / \
             value_returns_df['daily_return'].std()
    return sharpe

#------------------------------------------------------------------------------------------------
def run_model():

    # 1. Read the preprocessed working data from the file: We can move preprocessing to this code
    #    later but as of now we should be happy with this
    data  = pd.read_csv(WORKING_DATA_WITH_TE_PATH, parse_dates =['Date'])
    data["Date"] = [d.date() for d in data.Date]
    print("Working data with technical indicators details:")
    print("data.shape:{data.shape}")
    print(data.head())

    # 2. We already have data with complete set of records
    #    We will retrain our models after 60 business days
    model_retrain_dates = [ x[1] for x in enumerate(data.Date.unique()) if (x[0]+1)%RETRAIN_MODEL_CYCLE==0 ]

    value_returns_list = []
    weights_df_list=[]
    sharpe_a2c_list=[]

    trading_weights_list = []
    trading_value_returns_list = []

    model_retrain_dates_idx  = 0 # Just to have quick checks
    # VERSION 1. Lets live with one model only. Later we do ENSAMBLE
    for model_retrain_dates_idx in range(len(model_retrain_dates)):
        
        model_retrain_date = model_retrain_dates[model_retrain_dates_idx]
        print(f"Retraining model on {model_retrain_date}")

        # Out of this historical data we reserve last 60 days for validation
        # and any date before that training
        # The train data is from full history till the train_till_date
        train_till_date = model_retrain_date + timedelta(days=-VALIDATION_WINDOW)

        # Validation Range
        validation_from, validation_to = (train_till_date + timedelta(days=1), model_retrain_date)

        # 1. Training/validation data.
        train_data = data.loc[data.Date <= train_till_date]
        print(f"train_data.shape={train_data.shape}")
        validation_data = data.loc[(data.Date >= validation_from) & (data.Date <= validation_to)]
        print(f"validation_data.shape={validation_data.shape}")

        # 2. Params for environment
        params = {"investable_assets" : INVESTABLE_ASSETS, 
                 'initial_account_balance': 1e6, 
                 "save_debug_files":SAVE_DEBUG_FILES
                 }

        # A2C
        # 3. We train/validate the model
        params["train_mode"] = True        
        env_train = DummyVecEnv([lambda: StockEnv(train_data, params)])
        train_from_date_str = datetime.strftime(train_data.Date.values[1],"%Y%m%d")        
        train_till_date_str = datetime.strftime(train_data.Date.values[-1],"%Y%m%d")
        print(f"======A2C Training from:{train_from_date_str} to:{train_till_date_str}========")
        model_name = "A2C_multiasset_{}_{}".format(train_from_date_str,train_till_date_str)
        model_a2c = train_A2C(env_train, model_name, timesteps=25000)
        del env_train

        print(f"======A2C Validation from:{validation_from} to:{validation_to}========")
        params["train_mode"] = False
        env_val = DummyVecEnv([lambda: StockEnv(validation_data, params)])
        obs_val = env_val.reset()
        a2c_value_returns_df, a2c_weights_df = DRL_validation(model=model_a2c, test_data=validation_data, test_env=env_val, test_obs=obs_val)
        # - get sharpe ratio etc
        sharpe_a2c = get_sharpe(a2c_value_returns_df)
        #print(a2c_weights_df[a2c_weights_df.tic=="GLD"])
        print(f"sharpe_a2c:{sharpe_a2c}")
        del env_val

        # Unomment finally 
        """ # PPO
        print(f"======PPO Training from:{train_from_date_str} to:{train_till_date_str}========")
        params["train_mode"] = True
        model_name = "PPO_multiasset_{}_{}".format(train_from_date_str,train_till_date_str)
        env_train = DummyVecEnv([lambda: StockEnv(train_data, params)])
        model_ppo = train_PPO(env_train, model_name, timesteps=25000)
        del env_train

        print(f"======PPO Validation from:{validation_from} to:{validation_to}========")
        params["train_mode"] = False
        env_val = DummyVecEnv([lambda: StockEnv(validation_data, params)])
        obs_val = env_val.reset()
        ppo_value_returns_df, ppo_weights_df = DRL_validation(model=model_ppo, test_data=validation_data, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_sharpe(ppo_value_returns_df)
        print(f"sharpe_ppo:{sharpe_ppo}")
        del env_val

        # DDPG
        print(f"======DDPG Training from:{train_from_date_str} to:{train_till_date_str}========")
        params["train_mode"] = True
        model_name = "DDPG_multiasset_{}_{}".format(train_from_date_str,train_till_date_str)
        env_train = DummyVecEnv([lambda: StockEnv(train_data, params)])
        model_ddpg = train_DDPG(env_train, model_name, timesteps=25000)
        del env_train

        print(f"======DDPG Validation from:{validation_from} to:{validation_to}========")
        params["train_mode"] = False
        env_val = DummyVecEnv([lambda: StockEnv(validation_data, params)])
        obs_val = env_val.reset()
        ddpg_value_returns_df, ddpg_weights_df = DRL_validation(model=model_ddpg, test_data=validation_data, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_sharpe(ddpg_value_returns_df)
        print(f"sharpe_ddpg:{sharpe_ddpg}")
        del env_val """


        # TRADE Till next model_retrain_date. ie. backtest
        chosen_model = model_a2c

        trade_till_date = None
        if model_retrain_date == model_retrain_dates[-1]:
            # If last training then trade till last
            trade_till_date = data.Date.values[-1]
        else:
            # If not last training then trade till a day before next model training date
            trade_till_date = model_retrain_dates[model_retrain_dates_idx+1] + timedelta(days=-1)
        
        # Maximum date will which last trading happened, so that we seamlessly continue
        trading_start_date = max(data.Date[data.Date < model_retrain_date])
        trading_data = data.loc[(data.Date >= trading_start_date) & (data.Date <= trade_till_date) ]


        params["train_mode"] = False
        if model_retrain_dates_idx > 0:
            # Use customized weights and new $ amount
            # Use last trading weights
            last_weights = trading_weights_list[-1]
            params["custom_init_weight"] = dict(last_weights[["tic","new_weight"]].apply(lambda x: (x[0],x[1]), axis = 1).values)
            # Last cumulative value
            params['initial_account_balance'] = trading_value_returns_list[-1].account_value.values[-1]
        else:
            # default account balance to start
            params['initial_account_balance'] = 1000000.0
            

        env_trade = DummyVecEnv([lambda: StockEnv(trading_data, params)])
        obs_trade = env_trade.reset()
        # We run ne date less to test till last. The second last date is the termination date
        dates = trading_data.Date.unique()
        rewards, done, info = None, None, None
        for dt in dates[0:(len(dates)-1)]:
            action, _states = chosen_model.predict(obs_trade)
            obs_trade, rewards, done, info = env_trade.step(action)
            # save weights
            trading_weights_list.append(info[0]['weights'])
            # if it was terminal, save values of returns
            if done[0]:
                trading_value_returns_list.append(info[0]['value_returns_df'])

    # All dates loop ends here
    trading_weights_df = pd.concat(trading_weights_list)
    trading_value_returns_df  = pd.concat(trading_value_returns_list)
    print(sharpe_a2c_list)
    print("Writing trading_weights_df.csv")
    trading_weights_df.to_csv('results/trading_weights_df.csv', index=False)
    print("Writing trading_value_returns_df.csv")
    trading_value_returns_df.to_csv('results/trading_value_returns_df.csv', index=False)
    pass


if __name__ == "__main__":
    print("Running STONK trader model")

    try:
        run_model()
    except Exception as e:
        print(f"Got exception : {e}")
        raise e
    
    print("Finished STONK trader model.")



    

