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


class StockEnvTrain(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df, params):
        self._seed()
        self.df = df
        self.params = params

        # TBD: Check if this env is created for each day of training.
        # My guess is that it is created once for each simulation history
        # For intermediate calls only the step is called
        self.day = 0

        # These are must for any environment: terminal, state, reward, action_space,observation_space
        self.terminal = False
        
        # initialize state
        self.train_dates = self.df.Date.unique()
        data = self.df.loc[self.df.Date == self.train_dates[0]]

        # make these members to avoid recalculations later in step
        self.bonds = self.params["investable_assets"]["bonds"]
        self.equities = self.params["investable_assets"]["equities"]
        self.commodities = self.params["investable_assets"]["commodities"]
        self.all_tic = self.bonds + self.equities + self.commodities
        ordered_tmp = pd.DataFrame({"tic":self.all_tic}).merge(data[["tic","macd","rsi", "cci","adx","turbulence"]], 
                                            how="left", 
                                            left_on = ["tic"], 
                                            right_on = ["tic"])
        # second line weights/$: TBD
        self.state = [self.params['initial_account_balance']] + \
                      [0]*len(self.all_tic)  + \
                      ordered_tmp.macd.values.tolist() + \
                      ordered_tmp.rsi.values.tolist() + \
                      ordered_tmp.cci.values.tolist() + \
                      ordered_tmp.adx.values.tolist() + \
                      ordered_tmp.turbulence.values.tolist()
                      
        # initialize reward
        self.reward = 0
        self.cost = 0
        # action_space: original paper takes -1 to 1 which means sell 100% or buy 100%
        self.action_space = spaces.Box(low = -1, high = 1,shape = (len(self.all_tic),)) 
        # observation_space: same as self.state - should correspond one to one  
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (len(self.state),))
        # memorize all the total balance change
        self.asset_memory = [self.params['initial_account_balance']]
        self.rewards_memory = []
        pass

    def _sell_assets(self, action):
        #TODO
        1+2
        pass

    def _buy_assets(self, index, action):
        #TODO
        1+2
        pass

    def step(self, actions):
        #TODO
        1+2

        return self.state, self.reward, self.terminal,{}

    def reset(self):
        # Can we move this repeated code to a common reset. Same is used in the init
        self.day = 0        
        # initialize state
        data = self.df.loc[self.df.Date == self.train_dates[self.day]]
        # make these members to avoid recalculations later in step
        self.bonds = self.params["investable_assets"]["bonds"]
        self.equities = self.params["investable_assets"]["equities"]
        self.commodities = self.params["investable_assets"]["commodities"]
        self.all_tic = self.bonds + self.equities + self.commodities
        ordered_tmp = pd.DataFrame({"tic":self.all_tic}).merge(data[["tic","macd","rsi", "cci","adx","turbulence"]], 
                                            how="left", 
                                            left_on = ["tic"], 
                                            right_on = ["tic"])
        # second line weights/$: TBD
        self.state = [self.params['initial_account_balance']] + \
                      [0]*len(self.all_tic)  + \
                      ordered_tmp.macd.values.tolist() + \
                      ordered_tmp.rsi.values.tolist() + \
                      ordered_tmp.cci.values.tolist() + \
                      ordered_tmp.adx.values.tolist() + \
                      ordered_tmp.turbulence.values.tolist()
                      
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [self.params['initial_account_balance']]
        self.rewards_memory = []
        return self.state
    
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
    model_retrain_dates = [ x[1] for x in enumerate(data.Date) if (x[0]+1)%RETRAIN_MODEL_CYCLE==0 ]

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



    

