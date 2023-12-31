from network import FeedForwardNN
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np

class PPO:
    def __init__(self, env):
        self._init_hyperparameters()

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALG STEP 1
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1) 

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
    
    def get_action(self, obs):

        obs = torch.tensor(obs, dtype=torch.float)
        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def _init_hyperparameters(self):

        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5

        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005

    def rollout(self):
        # Batch data
        batch_obs = [] # (number of timesteps per batch, dimension of observation)         
        batch_acts = [] # (number of timesteps per batch, dimension of action)           
        batch_log_probs = [] # (number of timesteps per batch)      
        batch_rews = [] # (number of episodes, number of timesteps per episode)           
        batch_rtgs = [] # (number of timesteps per batch)           
        batch_lens = [] # (number of episodes)           

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()[0]
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, trunc, _ = self.env.step(action)
                done = done or trunc

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.critic(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def learn(self, total_timesteps):
        t_so_far = 0

        while t_so_far < total_timesteps:              # ALG STEP 2
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens \
                = self.rollout()
            
            t_so_far += np.sum(batch_lens)
            
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5
            A_k = batch_rtgs - V.detach()
            # advantage normalization
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # ALG STEP 6 & 7
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t), batch_log_probs is pi_old
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)  

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()


import gym
env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)