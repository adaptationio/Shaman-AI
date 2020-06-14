'''
A large part of the code in this file was sourced from the rl-baselines-zoo library on GitHub.
In particular, the library provides a great parameter optimization set for the PPO2 algorithm,
as well as a great example implementation using optuna.
Source: https://github.com/araffin/rl-baselines-zoo/blob/master/utils/hyperparams_opt.py
'''

import optuna

import pandas as pd
import numpy as np

from pathlib import Path
import time

import gym
import numpy as np
import os
import datetime
import csv
import argparse
from functools import partial
import time
from stable_baselines.common.policies import MlpLnLstmPolicy, LstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv,VecNormalize 
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2, SAC
from stable_baselines.deepq import DQN
#from stable_baselines.deepq.policies import FeedForwardPolicy
from ..env import Template_Gym
from ..common import CustomPolicy, CustomPolicy_2, CustomLSTMPolicy, CustomPolicy_4, CustomPolicy_3, CustomPolicy_5
from ..common import PairList, PairConfig, PairsConfigured
#env = Template_Gym()
#from stable_baselines.gail import generate_expert_traj

#from stable_baselines.gail import ExpertDataset


timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')
pc = PairsConfigured()
class Optimization():
    def __init__(self, config):

        self.reward_strategy = 'sortino2'
        #self.input_data_file = 'data/coinbase_hourly.csv'
        self.params_db_file = 'sqlite:///params.db'

        # number of parallel jobs
        self.n_jobs = 1
        # maximum number of trials for finding the best hyperparams
        self.n_trials = 1000
        #number of test episodes per trial
        self.n_test_episodes = 10
        # number of evaluations for pruning per trial
        self.n_evaluations = 10
        self.config = config

        #self.df = pd.read_csv(input_data_file)
        #self.df = df.drop(['Symbol'], axis=1)
        #self.df = df.sort_values(['Date'])
        #self.df = add_indicators(df.reset_index())

        #self.train_len = int(len(df) * 0.8)

        #self.df = df[:train_len]

        #self.validation_len = int(train_len * 0.8)
        #self.train_df = df[:validation_len]
        #self.test_df = df[validation_len:]

    def make_env(self, env_id, rank, seed=0, eval=False,config=pc.configeurcad4h):
        """
        Utility function for multiprocessed env.
    
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            self.config = config
            self.eval= eval
            env = Template_Gym(config=self.config, eval=self.eval)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    
    # Categorical parameter
    #optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])

    # Int parameter
    #num_layers = trial.suggest_int('num_layers', 1, 3)

    # Uniform parameter
    #dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

    # Loguniform parameter
    #learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Discrete-uniform parameter
    #drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)
    def optimize_envs(self, trial):
        return {
            'reward_func': self.reward_strategy,
            'forecast_len': int(trial.suggest_loguniform('forecast_len', 1, 200)),
            'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
        }

    def optimize_config(self, trial):
        return {
            'sl': trial.suggest_loguniform('sl', 1.0, 10.0),
            'tp': trial.suggest_loguniform('tp', 1.0 ,10.0)
            
        }

    def optimize_ppo2(self,trial):
        return {
            'n_steps': int(trial.suggest_int('n_steps', 16, 2048)),
            'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
            'noptepochs': int(trial.suggest_int('noptepochs', 1, 48)),
            'lam': trial.suggest_uniform('lam', 0.8, 1.)
        }

    def optimize_lstm(self, trial):
        return {
            'lstm': trial.suggest_categorical('optimizer', ['lstm', 'mlp'])
            
        }
    def ob_types(self, trial):
        return {
            'lstm': trial.suggest_categorical('optimizer', ['lstm', 'mlp'])
            
        }


    def optimize_agent(self,trial):
        #self.env_params = self.optimize_envs(trial)
        env_id = "default"+str()
        num_e = 1  # Number of processes to use
        #self.config_param = self.optimize_config(trial)
        #self.config.sl = self.config_param['sl']
        #self.config.sl = self.config_param['tp']
        #self.model_type = self.optimize_lstm(trial)
        #self.model_type = self.model_type['lstm']
        self.model_type = "mlp"
        if self.model_type == 'mlp':
            self.policy = CustomPolicy_5
        else:
             self.policy = CustomPolicy_4 
        self.train_env = SubprocVecEnv([self.make_env(env_id+str('train'), i, eval=False, config=self.config) for i in range(num_e)])
        #self.train_env = SubprocVecEnv([self.make_env(env_id, i, eval=False) for i in range(num_e)])
        self.train_env = VecNormalize(self.train_env, norm_obs=True, norm_reward=True)
        self.test_env =SubprocVecEnv([self.make_env(env_id+str("test"), i, eval=True, config=self.config) for i in range(num_e)])
        #self.test_env = SubprocVecEnv([self.make_env(env_id, i, eval=True) for i in range(num_e)])
        self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=True)
        try:
            self.test_env.load_running_average("saves")
            self.train_env.load_running_average("saves")
        except:
            print('cant load')
        self.model_params = self.optimize_ppo2(trial)
        self.model = PPO2(self.policy, self.train_env, verbose=0, nminibatches=1, tensorboard_log="./gbp_chf_single", **self.model_params )
        #self.model = PPO2(CustomPolicy_2, self.env, verbose=0, learning_rate=1e-4, nminibatches=1, tensorboard_log="./min1" )

        last_reward = -np.finfo(np.float16).max
        #evaluation_interval = int(len(train_df) / self.n_evaluations)
        evaluation_interval = 36525

        for eval_idx in range(self.n_evaluations):
            try:
                self.model.learn(evaluation_interval)
                self.test_env.save_running_average("saves")
                self.train_env.save_running_average("saves")
            except:
                print('did not work')

            rewards = []
            n_episodes, reward_sum = 0, 0.0
            print('Eval')
            obs = self.test_env.reset()
            #state = None
            #done = [False for _ in range(self.env.num_envs)]
            while n_episodes < self.n_test_episodes:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.test_env.step(action)
                reward_sum += reward

                if done:
                    rewards.append(reward_sum)
                    reward_sum = 0.0
                    n_episodes += 1
                    obs = self.test_env.reset()

            last_reward = np.mean(rewards)
            trial.report(-1 * last_reward, eval_idx)

            if trial.should_prune(eval_idx):
                raise optuna.structs.TrialPruned()

        return -1 * last_reward


    def optimize(self, config):
        self.config = config
        study_name = 'ppo2_single_ready'
        study_name = 'ppo2_single_ready_nosltp'
        study_name = 'ppo2_single_ready_nosltp_all_yeah'
        study_name = 'ppo2_eur_gbp_op'
        study_name = 'ppo2_gbp_chf_op'
        study_name = 'ppo2_gbp_chf_h1_new1'
        study_name = 'ppo2_gbp_chf_h4_r_new11'
        study_name = 'ppo2_gbp_chf_h4_r_withvolfixed'
        study_name = 'ppo2_gbp_chf_h4_r_withvolclosefix212'
        study_name = 'ppo2_gbp_chf_h4_loged_sortinonew'
        study = optuna.create_study(
            study_name=study_name, storage=self.params_db_file, load_if_exists=True)

        try:
            study.optimize(self.optimize_agent, n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        print('Number of finished trials: ', len(study.trials))

        print('Best trial:')
        trial = study.best_trial

        print('Value: ', trial.value)

        print('Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        return study.trials_dataframe()


#if __name__ == '__main__':
    #optimize()