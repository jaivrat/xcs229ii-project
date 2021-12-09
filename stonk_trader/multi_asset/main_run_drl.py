# common library
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

from stable_baselines.common.vec_env import DummyVecEnv
import gym
from gym.utils import seeding
from gym import spaces

# RL models from stable-baselines
from stable_baselines import A2C


# Working data with technical indicators
WORKING_DATA_WITH_TE_PATH = "data/wd_te.csv"
# We will retrain our models after 60 business days
RETRAIN_MODEL_CYCLE = 60
# Investable assets 
INVESTABLE_ASSETS = dict(# Bonds
                          bonds = ["LQD", "SHY", "IEF", "TLT", "AGG"], 
                          # Equities  
                          equities = ["IJH", "IJR", "IVV", "IVE", "IVW","SPY"], # "^GSPC"
                          # Commodities
                          commodities = ["GLD"])

# TRANSACTION FEE
TRANSACTION_FEE_PERCENT = 0.001
# Gamma or reward scaling 
REWARD_SCALING = 1e-4

class StockEnvTrain(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df, params):
        # TBD: Check if this env is created for each day of training.
        # My guess is that it is created once for each simulation history
        # For intermediate calls only the step is called
        self._seed()
        self.params = params

        # make these members to avoid recalculations later in step
        self.num_asset_classes = len(self.params["investable_assets"].keys())
        self.bonds = self.params["investable_assets"]["bonds"]
        self.equities = self.params["investable_assets"]["equities"]
        self.commodities = self.params["investable_assets"]["commodities"]
        self.investable_tic = self.bonds + self.equities + self.commodities

        self.df = df.copy()
        self.train_dates = self.df.Date.unique()
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

    def _common_init_reset(self):
        self.day = 0
        # These are must for any environment: terminal, state, reward, action_space,observation_space
        self.terminal = False        

        self.data = self.df.loc[self.df.Date == self.train_dates[0]]
        # second line weights/$: TBD
        bond_weight = 0.40
        equity_weight = 0.60
        commodity_weight = 0.00
        # further distribution of state
        bonds_distr_wt = [1.0/len(self.bonds) for _ in self.bonds]
        equity_distr_wt = [1.0/len(self.equities) for _ in self.equities]
        commodity_distr_wt = [1.0/len(self.commodities) for _ in self.commodities]
        # state:
        # portfolio $ value => 1
        # [bond_weight, equity_weight,  commodity_weight] => 3
        # bonds_distr_wt + equity_distr_wt + commodity_distr_wt => #bonds + #equities + #commodities
        # close price values => self.data.shape[0]
        # macd.values.tolist() => self.data.shape[0]
        # rsi.values.tolist() => self.data.shape[0]
        # cci.values.tolist() => self.data.shape[0]
        # adx.values.tolist() => self.data.shape[0]
        # turbulence.values.tolist()=> 1  # only one turbulence value
        self.state =  [self.params['initial_account_balance']]+\
                      [bond_weight, equity_weight, commodity_weight] +\
                      bonds_distr_wt + equity_distr_wt + commodity_distr_wt + \
                      self.data.close.values.tolist() +\
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() + \
                      [self.data.turbulence.values[0]]

        # Store once to quickly access later in step
        self.state_offset_dict = dict( dollar_position = 0,
                                       asset_class_weights = (1, 1+ self.num_asset_classes),
                                       asset_distr_weights = (1+ self.num_asset_classes, 1+ self.num_asset_classes + len(self.investable_tic)),
                                       close = (1+ self.num_asset_classes + len(self.investable_tic), 
                                               1 + self.num_asset_classes + len(self.investable_tic) + self.data.shape[0])                    
                                     )
                      
        # initialize reward
        self.reward = 0
        self.cost = 0
        # action_space: original paper takes -1 to 1 which means sell 100% or buy 100%
        # here we take 3 level of actions. 
        # First: set 3 elements are proportional weights of bond, equity and commodity
        # Second: next bonds_distr_wt + equity_distr_wt + commodity_distr_wt
        self.action_space = spaces.Box(low = 0.00000001, high = 1, shape = (self.num_asset_classes + len(self.investable_tic),)) 
        # observation_space: same as self.state - should correspond one to one  
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (len(self.state),))
        # memorize all the total balance change
        self.asset_memory = [self.params['initial_account_balance']]
        self.rewards_memory = []

    def _sell_assets(self, action):
        #TODO
        1+2
        pass

    def _buy_assets(self, index, action):
        #TODO
        1+2
        pass

    def step(self, actions):
        
        # Terminal state is market if it is last day of training set
        self.terminal = (self.day == len(self.train_dates)-1)

        if self.terminal:   
            # This is terminal date
            print("Comes to Terminal")

            
            pass
        else:
            if np.isnan(actions).any():
                print("DEBUG: check why action is nan")
            if self.day == 51:
                print("DEBUG: check why action becomes nan later")
            # This is not terminal date
            # beginning of period assets 
            n_assets = len(self.investable_tic)
            asset_class_weights = self.state[self.state_offset_dict['asset_class_weights'][0]:self.state_offset_dict['asset_class_weights'][1]]
            asset_class_dist_weights = self.state[self.state_offset_dict['asset_distr_weights'][0]:self.state_offset_dict['asset_distr_weights'][1]]
            asset_close_prices  = self.state[self.state_offset_dict['close'][0]:self.state_offset_dict['close'][1]]


            # -- bond weights
            bond_distr = np.array(asset_class_dist_weights[0:len(self.bonds)])
            bond_distr = bond_distr/sum(bond_distr)
            # -- equity weights
            equity_distr = np.array(asset_class_dist_weights[len(self.bonds):len(self.bonds)+len(self.equities)])
            equity_distr = equity_distr/sum(equity_distr)
            # -- commodity weights
            commod_distr = np.array(asset_class_dist_weights[(len(self.bonds)+len(self.equities)):(len(self.bonds)+len(self.equities)+len(self.commodities))])
            commod_distr = commod_distr/sum(commod_distr)

            self.day += 1
            self.data = self.df.loc[self.df.Date == self.train_dates[self.day]]

            investable_prev_weights = (asset_class_weights[0] *bond_distr).tolist() + \
                                        (asset_class_weights[1] *equity_distr).tolist() + \
                                        (asset_class_weights[2] *commod_distr).tolist()
            investable_returns = self.data.returns_close.values[0:len(self.investable_tic)] # note that it is already ordered in tic seq


            # Action weight:
            act_asset_class_weights = actions[0:self.num_asset_classes]
            act_asset_class_weights = act_asset_class_weights/sum(act_asset_class_weights)
            # bonds
            act_bond_distr = actions[self.num_asset_classes:(self.num_asset_classes +len(self.bonds))]
            # equities
            act_eq_distr = actions[ self.num_asset_classes + len(self.bonds):self.num_asset_classes + len(self.bonds)+ len(self.equities)]
            # commodities
            act_commod_dist = actions[self.num_asset_classes + len(self.bonds)+ len(self.equities): 
                                      self.num_asset_classes + len(self.bonds)+ len(self.equities) + len(self.commodities)]
            # Normalize
            act_bond_distr = act_bond_distr/sum(act_bond_distr)
            act_eq_distr = act_eq_distr/sum(act_eq_distr)
            act_commod_dist = act_commod_dist/sum(act_commod_dist)

            investable_new_weights = np.array((act_asset_class_weights[0] *act_bond_distr).tolist() + \
                                        (act_asset_class_weights[1] *act_eq_distr).tolist() + \
                                        (act_asset_class_weights[2] *act_commod_dist).tolist())


            # Total return
            total_returns = sum(investable_prev_weights * investable_returns)
            begin_total_asset = self.state[0]
            end_total_asset = begin_total_asset * ( 1.0 + total_returns)
            
            pre_rebal_pos = np.array(investable_prev_weights) * begin_total_asset * (1.0 + investable_returns)
            new_desired_rebal_pos = investable_new_weights * end_total_asset
            transaction_fee = sum(np.abs(pre_rebal_pos - new_desired_rebal_pos)*TRANSACTION_FEE_PERCENT)

            self.reward = (end_total_asset - begin_total_asset - transaction_fee)
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING


            # We also need to update new state
            self.state =  [end_total_asset]+\
                          act_asset_class_weights.tolist() +\
                      act_bond_distr.tolist() + act_eq_distr.tolist() + act_commod_dist.tolist() + \
                      self.data.close.values.tolist() +\
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() + \
                      [self.data.turbulence.values[0]]


        return self.state, self.reward, self.terminal,{}

    
    
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
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


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

    # VERSION 1. Lets live with one model only. Later we do ENSAMBLE
    for model_retrain_date in model_retrain_dates:
        print(f"Retraining model on {model_retrain_date}")

        # 1. select training data. The data is from full history till the model_retrain_date
        train_data = data.loc[data.Date <= model_retrain_date]
        print(train_data.shape)

        # 2. Create the training env
        params = {"investable_assets" : INVESTABLE_ASSETS, 'initial_account_balance': 1e6}
        env_train = DummyVecEnv([lambda: StockEnvTrain(train_data, params)])

        # 3. We train the model
        train_from_date_str = datetime.strftime(train_data.Date.values[1],"%Y%m%d")        
        train_till_date_str = datetime.strftime(train_data.Date.values[-1],"%Y%m%d")
        model_name = "A2C_multiasset_{}_{}".format(train_from_date_str,train_till_date_str)
        model_a2c = train_A2C(env_train, model_name, timesteps=25000)


        1 + 2

    pass



if __name__ == "__main__":
    print("Running STONK trader model")

    try:
        run_model()
    except Exception as e:
        print(f"Got exception : {e}")
        raise e
    
    print("Finished STONK trader model.")



    

