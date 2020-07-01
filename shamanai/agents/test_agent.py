import time

import gym
import numpy as np
import os
import datetime
import csv
import argparse
from functools import partial
import time
import datetime
import pytz
from stable_baselines.common.policies import MlpLnLstmPolicy, LstmPolicy, CnnPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv,VecNormalize 
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2, SAC
from stable_baselines.deepq import DQN
#from stable_baselines.deepq.policies import FeedForwardPolicy
from .config import Config
from ..env import Template_Gym
from .human_agent_TB import HumanAgentTB
from .human_agent_RT import HumanAgentRT

#env = Template_Gym()
from stable_baselines.gail import generate_expert_traj

from stable_baselines.gail import ExpertDataset


timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')
config = Config()
class PPO2_SB_TEST():
    def __init__(self):
        self.love = 'Ramona'
        self.env_fns = [] 
        self.env_names = []
        self.config = Config()
    
    def make_env(self, env_id, rank, seed=0, eval=False,config=config):
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
            env = Template_Gym(config=self.config)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    
    def train(self, num_e=1, n_timesteps=1000000, save_fraction=0.0125, save='saves/audbuyh4120', config=config):
        env_id = "default"
        num_e = 1  # Number of processes to use
        # Create the vectorized environment
        #env = DummyVecEnv([lambda: env])
        #Ramona
        self.config = config
        self.env = SubprocVecEnv([self.make_env(env_id, i, eval=False, config=self.config) for i in range(num_e)])
        #env = Template_Gym()
        #self.env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(self.env, norm_obs=False, norm_reward=True)
        self.model = PPO2(CnnPolicy, self.env, verbose=0)
        #self.model = PPO2("MlpPolicy", self.env, verbose=0, nminibatches=1, tensorboard_log="./aud_chf", learning_rate=1e-5  )
        #self.model = PPO2(CustomPolicy_4, self.env, verbose=0, nminibatches=1, tensorboard_log="./gbp_chf_4h_r", **self.config.params )
        #self.model = PPO2(CustomPolicy_5, self.env, verbose=0, nminibatches=1, tensorboard_log="./aud_chf", learning_rate=1e-5  )#**self.config.params
        #self.model = PPO2.load('saves/playerdetails39', self.env, policy=CustomPolicy,  tensorboard_log="./playerdetailsex" )
        #self.model = PPO2.load(self.config.path+str(79)+'.pkl', self.env, policy=CustomPolicy_5,  tensorboard_log="./default/" )
        #self.model = PPO2.load("default9", self.env, policy=CustomPolicy, tensorboard_log="./test/" )
        n_timesteps = n_timesteps * save_fraction
        n_timesteps = int(n_timesteps)
        training_loop = 1 / save_fraction
        training_loop = int(training_loop)
        log_dir = "saves"
        #self.env.load_running_average(log_dir)
        for i in range(training_loop):
            self.model.learn(n_timesteps)
            self.model.save(self.config.save+str(i))
            #self.env.save_running_average(log_dir)
        #self.env.save_running_average(log_dir)
    
    
    def evaluate(self, num_env=1, num_steps=101, load='saves/audbuyh1', runs=80, config=config):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward
        """
        env_id = "moose"
        num_e = 1
        self.config = config
        log_dir = "moose"
        
        #log_dir = self.config.norm
        #self.env = SubprocVecEnv([self.make_env(env_id, i, eval=True) for i in range(num_env)])
        self.env = SubprocVecEnv([self.make_env(env_id, i, eval=True, config=self.config) for i in range(num_env)])
        #self.model = PPO2(CustomPolicy, self.env, verbose=1, learning_rate=1e-5, tensorboard_log="./default" )
        self.env = VecNormalize(self.env, norm_obs=False, norm_reward=True)
        try:
            self.env.load_running_average(log_dir)
        except:
            print('cant load')
        for i in range(runs):
            #self.model = PPO2(CustomPolicy, self.env, verbose=0, learning_rate=1e-5, tensorboard_log="./moose14" )
            #self.model = PPO2.load(self.config.path, self.env, policy=CustomPolicy_2,  tensorboard_log="./default/" )
            self.model = PPO2.load(self.config.save+str(77), self.env, policy=MlpPolicy,  tensorboard_log="./default/" )
            #self.env.load_running_average(log_dir)
            episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
            #self.total_pips = []
            obs = self.env.reset()
            state = None
            # When using VecEnv, done is a vector
            done = [False for _ in range(self.env.num_envs)]
            for i in range(num_steps):
                # _states are only useful when using LSTM policies
                action, state = self.model.predict(obs, state=state, mask=done, deterministic=True)
                obs, rewards , dones, _ = self.env.step(action)
                #actions, _states = self.model.predict(obs)
                # # here, action, rewards and dones are arrays
                 # # because we are using vectorized env
                #obs, rewards, dones, info = self.env.step(actions)
                #self.total_pips.append(self.env.player.placement)
        
        # Stats
                for i in range(self.env.num_envs):
                    episode_rewards[i][-1] += rewards[i]
                    if dones[i]:
                        episode_rewards[i].append(0.0)
            #self.env.save_running_average(log_dir)
            mean_rewards =  [0.0 for _ in range(self.env.num_envs)]
            n_episodes = 0
            for i in range(self.env.num_envs):
                mean_rewards[i] = np.mean(episode_rewards[i])     
                n_episodes += len(episode_rewards[i])   

        # Compute mean reward
            mean_reward = np.mean(mean_rewards)
            print("Mean reward:", mean_reward, "Num episodes:", n_episodes)
            #self.env.save(log_dir)

        return mean_reward

    def live(self, num_env=1, num_steps=1461, load='saves/gbp_usd_buy', runs=1, config=config):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward
        """
        self.config = config
        env_id = self.config.pair
        num_e = 1
        log_dir = self.config.log
       
        self.config.live = True
        self.config.load = False

        
        self.env = SubprocVecEnv([self.make_env(env_id, i, eval=True, config=self.config) for i in range(num_env)])
        
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        try:
            self.env.load_running_average(self.config.log)
        except:
            print('cant load')
            
        self.env.num_envs = 1
        for i in range(runs):
            self.model = PPO2.load(self.config.path+str(self.config.best)+'.pkl', self.env, policy=CustomPolicy_5,  tensorboard_log="./default/" )
            
            episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
            
            print(datetime.datetime.now())
            print(time.ctime())
            print('Market Check')
            for k in range(144000):
                times = datetime.datetime.now(tz=pytz.timezone('US/Eastern'))
                hour = int(times.hour)
                #minute = int(times[14:16])
                date = time.ctime()
                day = str(times.day)
                #print(day)
            
                if day == "Fri" and hour >=int(17) or day == "Sat" or day == "Sun" and hour <=int(17):
                    print('Market Closed')
                    time.sleep(int(60))
                        
                else:
                    print('Market Open')
                    break

            print("Market time check")
            for m in range(14400):
                times = str(datetime.datetime.now(tz=pytz.timezone('US/Eastern')))
                hour = int(times[11:13])
                minute = int(times[14:16])
                date = time.ctime()
                day = str(date[0:3])
                     
                if hour == 5 and minute == 1 or hour == 9 and minute==1 or hour == 13 and minute==1 or hour == 17 and minute ==1 or hour == 21 and minute ==1 or hour == 1 and minute == 1:
                    print(datetime.datetime.now())
                    break
                elif hour == 5 and minute == 2 or hour == 9 and minute==2 or hour == 13 and minute==2 or hour == 17 and minute ==2 or hour == 21 and minute ==2 or hour == 1 and minute == 2:
                    print(datetime.datetime.now())
                    break
                elif hour == 5 and minute == 3 or hour == 9 and minute==3 or hour == 13 and minute==3 or hour == 17 and minute ==3 or hour == 21 and minute ==3 or hour == 1 and minute == 3:
                    print(datetime.datetime.now())
                    break
                elif hour == 5 and minute == 4 or hour == 9 and minute==4 or hour == 13 and minute==4 or hour == 17 and minute ==4 or hour == 21 and minute ==4 or hour == 1 and minute == 4:
                    print(datetime.datetime.now())
                    break
                elif hour == 5 and minute == 5 or hour == 9 and minute==5 or hour == 13 and minute==5 or hour == 17 and minute ==5 or hour == 21 and minute ==5 or hour == 1 and minute == 5:
                    print(datetime.datetime.now())
                    break    
                else:
                    time.sleep(int(60))
            obs = self.env.reset()
            
            state = None
            
            # When using VecEnv, done is a vector
            done = [False for _ in range(self.env.num_envs)]
            
            for i in range(num_steps):

                
                # _states are only useful when using LSTM policies
                print("live step")
                action, state = self.model.predict(obs, state=state, mask=done, deterministic=True)
                obs, rewards , dones, _ = self.env.step(action)
        
                # # here, action, rewards and dones are arrays
                 # # because we are using vectorized env
                #obs, rewards, dones, info = self.env.step(actions)
                #self.total_pips.append(self.env.player.placement)
      
        # Stats
                for i in range(self.env.num_envs):
                    episode_rewards[i][-1] += rewards[i]
                    if dones[i]:
                        episode_rewards[i].append(0.0)
                    

                
                
                print(datetime.datetime.now())
                print(time.ctime())
                
                

            mean_rewards =  [0.0 for _ in range(self.env.num_envs)]
            n_episodes = 0
            

            for i in range(self.env.num_envs):
                mean_rewards[i] = np.mean(episode_rewards[i])     
                n_episodes += len(episode_rewards[i])  
                

        # Compute mean reward
            mean_reward = np.mean(mean_rewards)
            print("Mean reward_gbp_buy:", mean_reward, "Num episodes:", n_episodes)
            

        return mean_reward

    # The two function below are not working atm
    def pre_train(self, num_e=1, load="saves/m19"):
        env_id = 'default'
        num_e = 1
        log_dir = "saves"
        # Usingenv = make_env() only one expert trajectory
        # you can specify `traj_limitation=-1` for using the whole dataset
        dataset = ExpertDataset(expert_path='default2.npz',traj_limitation=1, batch_size=128)
        self.env = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_e)])
        #env = Template_Gym()
        #self.env = DummyVecEnv([lambda: env])
        #self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        #env = make_env()
        #model = GAIL("MlpPolicy", env=env, expert_dataset=dataset, verbose=1)
        #self.env.save_running_average("saves"+self.config.pair)
        self.model = PPO2(MlpPolicy, self.env, verbose=1, nminibatches=1,  learning_rate=1e-5, tensorboard_log="./m1ln4" )
        #self.model = PPO2.load("saves/m19", self.env, policy=CustomPolicy, tensorboard_log="./default/" )
        #self.env.save_running_average("saves"+self.config.pair)
        # Pretrain the PPO2 model
        self.model.pretrain(dataset, n_epochs=10000)

        # As an option, you can train the RL agent
        #self.model.learn(int(100000000))

        # Test the pre-trained model
        self.env = self.model.get_env()
        #self.env.save_running_average("saves"+self.config.pair)
        obs = self.env.reset()

        reward_sum = 0.0
        for _ in range(11):
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = self.env.step(action)
            reward_sum += reward
            #self.env.render()
            if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = self.env.reset()

        self.env.close()



    def gen_pre_train(self, num_e=1, save='default2', episodes=10):
        #self.create_envs(game_name=game, state_name=state, num_env=num_e)
        #self.env=SubprocVecEnv(self.env_fns)
        env_id = 'default'
        num_e = 1
        self.env = Template_Gym(config=self.config)
        #env = Template_Gym()
        #self.env = DummyVecEnv([lambda: env])
        #self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        #env = make_env()
        #model = GAIL("MlpPolicy", env=env, expert_dataset=dataset, verbose=1)
        #self.env.load_running_average("saves")
        #self.model = PPO2.load("saves/m19", self.env, policy=CustomPolicy, tensorboard_log="./default/" )
        #self.env.load_running_average("saves")
        #env = make_env()
        #self.expert_agent = 
        #self.model = HumanAgentTB()
        self.model = HumanAgentRT()
        generate_expert_traj(self.model.train, save, self.env, n_episodes=episodes)
        

