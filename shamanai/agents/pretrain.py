def pre_train(self):
        # Using only one expert trajectory
        # you can specify `traj_limitation=-1` for using the whole dataset
        dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                                                                                                traj_limitation=1, batch_size=128)

        model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
        # Pretrain the PPO2 model
        model.pretrain(dataset, n_epochs=1000)

        # As an option, you can train the RL agent
        # model.learn(int(1e5))

        # Test the pre-trained model
        env = model.get_env()
        obs = env.reset()

        reward_sum = 0.0
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = env.reset()

        env.close()

    def gen_pre_train(self, game, state, num_e=1, save='default2', episodes=10):
        self.create_envs(game_name=game, state_name=state, num_env=num_e)
        env=SubprocVecEnv(self.env_fns)
        self.expert_agent = "moose"
        self.generate_expert_traj(self.expert_agent, save, env, n_episodes=episodes)