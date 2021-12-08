# common library
import pandas as pd
import numpy as np
import os
import time

from stable_baselines.common.vec_env import DummyVecEnv
import gym
from gym.utils import seeding
from gym import spaces


# Working data with technical indicators
WORKING_DATA_WITH_TE_PATH = "data/wd_te.csv"
# We will retrain our models after 60 business days
RETRAIN_MODEL_CYCLE = 60


class StockEnvTrain(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        self._seed()
        self.df = df

        # These are must for any environment: terminal, state, reward, action_space,observation_space
        self.terminal = False
        # initialize state
        self.state = [0]
        # initialize reward
        self.reward = 0
        # action_space => set this properly
        self.action_space = spaces.Box(low = -1, high = 1,shape = (2,)) 
        # observation_space => set this properly
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (181,))
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
        #TODO
        1+2
        self.terminal = False
        return self.state
    
    def render(self, mode='human'):
        1+2
        return self.state

    def _seed(self, seed=None):
        1+2
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



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
        env_train = DummyVecEnv([lambda: StockEnvTrain(train_data)])

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



    

