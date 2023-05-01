# import copy
# import pickle
# import random
# import gymnasium as gym
# import torch
# from collections import deque, namedtuple
# from gymnasium.utils.save_video import save_video
# from torch import nn
# from torch.optim import Adam
# from torch.distributions import Categorical
# from utils import *


# # Class for training an RL agent with Actor-Critic
# class ACTrainer:
#     def __init__(self, params):
#         self.params = params
#         self.env = gym.make(self.params['env_name'])
#         self.agent = ACAgent(env=self.env, params=self.params)
#         self.actor_net = ActorNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
#         self.critic_net = CriticNet(input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
#         self.actor_optimizer = Adam(params=self.actor_net.parameters(), lr=self.params['actor_lr'])
#         self.critic_optimizer = Adam(params=self.critic_net.parameters(), lr=self.params['critic_lr'])
#         self.trajectory = None

#     def run_training_loop(self):
#         list_ro_reward = list()
#         for ro_idx in range(self.params['n_rollout']):
#             self.trajectory = self.agent.collect_trajectory(policy=self.actor_net)
#             self.update_critic_net()
#             self.estimate_advantage()
#             self.update_actor_net()
#             # TODO: Calculate avg reward for this rollout
#             # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
#             avg_ro_reward = sum(sum(traj_rewards) for traj_rewards in self.trajectory['reward']) / self.params['n_trajectory_per_rollout']
#             print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
#             # Append average rollout reward into a list
#             list_ro_reward.append(avg_ro_reward)
#         # Save avg-rewards as pickle files
#         pkl_file_name = self.params['exp_name'] + '.pkl'
#         with open(pkl_file_name, 'wb') as f:
#             pickle.dump(list_ro_reward, f)
#         # Save a video of the trained agent playing
#         self.generate_video()
#         # Close environment
#         self.env.close()

#     def update_critic_net(self):
#         for critic_iter_idx in range(self.params['n_critic_iter']):
#             self.update_target_value()
#             for critic_epoch_idx in range(self.params['n_critic_epoch']):
#                 critic_loss = self.estimate_critic_loss_function()
#                 critic_loss.backward()
#                 self.critic_optimizer.step()
#                 self.critic_optimizer.zero_grad()

#     def update_target_value(self, gamma=0.99):
#         # TODO: Update target values
#         # HINT: Use definition of target-estimate from equation 7 of teh assignment PDF
#         # states = self.trajectory['state']
#         # rewards = self.trajectory['reward']
#         # done = self.trajectory['done']

#         # next_state = self.trajectory['state'][1:] + self.trajectory['state'][:-1]

#         # with torch.no_grad():
#         #     tar_val = []
#         #     for i in range(len(states)):
#         #         if done[id]:
#         #             tar_val.append(rewards[i])
#         #         else:
#         #             tar_val.append(rewards[i] + gamma * self.critic_net(torch.tensor(next_state[i], dtype = torch.float32, device = get_device())).squeeze().item())

#         # self.trajectory['state_value'] = tar_val[:-1]
#         # self.trajectory['target_value'] = tar_val
#         state_values = []
#         for state in self.trajectory['obs']:
#             state_tensor = state.to(get_device())
#             state_value = self.critic_net(state_tensor).squeeze().detach().cpu().numpy()
#             state_values.append(state_value)
#         self.trajectory['state_value'] =    np.concatenate(state_values)
#         self.trajectory['target_value'] = [reward + gamma * next_value for reward, next_value in zip(self.trajectory['reward'][:-1], self.trajectory['state_value'][1:])]        

#     def estimate_advantage(self, gamma=0.99):
#         # TODO: Estimate advantage
#         # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF
#         advantages = []
#         rewards = self.trajectory['reward']
#         state_values = self.trajectory['state_value']
#         target_values = self.trajectory['target_value']

#         for t in range(len(rewards)):
#             # Compute the sum of discounted future rewards from time-step t
#             future_rewards = rewards[t:]
#             future_discounts = [gamma ** i for i in range(len(future_rewards))]
#             future_reward_sum = sum([a * b for a, b in zip(future_rewards, future_discounts)])

#             # Compute the sum of discounted future state-values from time-step t+1
#             future_values = state_values[t+1:]
#             future_value_discounts = [gamma ** i for i in range(len(future_values))]
#             future_value_sum = sum([a * b for a, b in zip(future_values, future_value_discounts)])

#             # Compute the target value for time-step t
#             target_value = rewards[t] + gamma * future_value_sum

#             # Compute the advantage for time-step t
#             advantage = target_value - state_values[t]
#             advantages.append(advantage)

#         self.trajectory['advantage'] = advantages

#     def update_actor_net(self):
#         actor_loss = self.estimate_actor_loss_function()
#         actor_loss.backward()
#         self.actor_optimizer.step()
#         self.actor_optimizer.zero_grad()

#     def estimate_critic_loss_function(self):
#         # TODO: Compute critic loss function
#         # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.
#         updated_size = max(len(self.trajectory['state_value']), len(self.trajectory['target_value'][0]))
#         state_value_padded = np.pad(self.trajectory['state_value'], (0, updated_size - len(self.trajectory['state_value'])), mode='constant', constant_values=0)
#         tar_value = np.pad(self.trajectory['target_value'][0], (0, updated_size - len(self.trajectory['target_value'][0])), mode='constant', constant_values=0)
#         #tar_val = torch.tensor(self.trajectory['target_value'], dtype=torch.float32, device=get_device())
#         #state_val = self.critic_net(torch.tensor(self.trajectory['state'], dtype=torch.float32, device=get_device())).squeeze()
#         critic_loss = nn.MSELoss()(torch.stack(tuple(torch.tensor(self.trajectory['state_value'],device=get_device()))), torch.tensor(tar_value, device=get_device()))
#         return critic_loss

#     def estimate_actor_loss_function(self):
#         actor_loss = list()
#         gamma = 0.99
#         for t_idx in range(self.params['n_trajectory_per_rollout']):
#             advantage = apply_discount(self.trajectory['advantage'][t_idx])
#             # TODO: Compute actor loss function
#             actions = self.trajectory['action'][t_idx]
#             log_probs = self.actor_net(torch.tensor(self.trajectory['state'][t_idx], dtype=torch.float32, device=get_device()))
#             log_probs = log_probs.gather(1, torch.tensor(actions, device=get_device()).view(-1, 1))
#             ratio = torch.exp(log_probs - torch.tensor(self.trajectory['log_prob'][t_idx], dtype=torch.float32, device=get_device()))
#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1 - self.params['eps_clip'], 1 + self.params['eps_clip']) * advantage
#             actor_loss.append(-torch.min(surr1, surr2).mean())

#         actor_loss = torch.stack(actor_loss).mean()

#         return actor_loss

#     def generate_video(self, max_frame=1000):
#         self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
#         obs, _ = self.env.reset()
#         for _ in range(max_frame):
#             action_idx, log_prob = self.actor_net(torch.tensor(obs, dtype=torch.float32, device=get_device()))
#             obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
#             if terminated or truncated:
#                 break
#         save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


# # CLass for actor-net
# class ActorNet(nn.Module):
#     def __init__(self, input_size, output_size, hidden_dim):
#         super(ActorNet, self).__init__()
#         # TODO: Define the actor net
#         # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
#         self.ff_net = nn.Sequential(
#             nn.Linear(input_size, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_size),
#             nn.Softmax(dim=-1)
#         )


#     def forward(self, obs):
#         # TODO: Forward pass of actor net
#         # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
#         act = self.ff_net(obs)
#         dist = Categorical(act)
#         action_index = dist.sample()
#         log_prob = dist.log_prob(action_index)
#         return action_index, log_prob


# # CLass for critic-net
# class CriticNet(nn.Module):
#     def __init__(self, input_size, output_size, hidden_dim):
#         super(CriticNet, self).__init__()
#         # TODO: Define the critic net
#         # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
#         self.ff_net = nn.Sequential(
#             nn.Linear(input_size, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_size)
#         )

#     def forward(self, obs):
#         # TODO: Forward pass of critic net
#         # HINT: (get state value from the network using the current observation)
#         state_value = self.ff_net(obs)
#         return state_value


# # Class for agent
# class ACAgent:
#     def __init__(self, env, params=None):
#         self.env = env
#         self.params = params
#         self.action_space = [action for action in range(self.env.action_space.n)]

#     def collect_trajectory(self, policy):
#         obs, _ = self.env.reset(seed=self.params['rng_seed'])
#         rollout_buffer = list()
#         for _ in range(self.params['n_trajectory_per_rollout']):
#             trajectory_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
#             while True:
#                 obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
#                 # Save observation
#                 trajectory_buffer['obs'].append(obs)
#                 action_idx, log_prob = policy(obs)
#                 obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
#                 # Save log-prob and reward into the buffer
#                 trajectory_buffer['log_prob'].append(log_prob)
#                 trajectory_buffer['reward'].append(reward)
#                 # Check for termination criteria
#                 if terminated or truncated:
#                     obs, _ = self.env.reset()
#                     rollout_buffer.append(trajectory_buffer)
#                     break
#         rollout_buffer = self.serialize_trajectory(rollout_buffer)
#         return rollout_buffer

#     # Converts a list-of-dictionary into dictionary-of-list
#     @staticmethod
#     def serialize_trajectory(rollout_buffer):
#         serialized_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
#         for trajectory_buffer in rollout_buffer:
#             serialized_buffer['obs'].append(torch.stack(trajectory_buffer['obs']))
#             serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
#             serialized_buffer['reward'].append(trajectory_buffer['reward'])
#         return serialized_buffer


# class DQNTrainer:
#     def __init__(self, params):
#         self.params = params
#         self.env = gym.make(self.params['env_name'])
#         self.q_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
#         self.target_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
#         self.target_net.load_state_dict(self.q_net.state_dict())
#         self.epsilon = self.params['init_epsilon']
#         self.optimizer = Adam(params=self.q_net.parameters(), lr=self.params['lr'])
#         self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

#     def run_training_loop(self):
#         list_ep_reward = list()
#         obs, _ = self.env.reset(seed=self.params['rng_seed'])
#         for idx_episode in range(self.params['n_episode']):
#             ep_len = 0
#             while True:
#                 ep_len += 1
#                 action = self.get_action(obs)
#                 next_obs, reward, terminated, truncated, info = self.env.step(action)
#                 if terminated or truncated:
#                     self.epsilon = max(self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
#                     next_obs = None
#                     self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
#                     list_ep_reward.append(ep_len)
#                     print(f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
#                     obs, _ = self.env.reset()
#                     break
#                 self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
#                 obs = copy.deepcopy(next_obs)
#                 self.update_q_net()
#                 self.update_target_net()
#         # Save avg-rewards as pickle files
#         pkl_file_name = self.params['exp_name'] + '.pkl'
#         with open(pkl_file_name, 'wb') as f:
#             pickle.dump(list_ep_reward, f)
#         # Save a video of the trained agent playing
#         self.generate_video()
#         # Close environment
#         self.env.close()

#     def get_action(self, obs):
#         # TODO: Implement the epsilon-greedy behavior
#         # HINT: The agent will will choose action based on maximum Q-value with
#         # '1-ε' probability, and a random action with 'ε' probability.
#         if np.random.uniform() < self.epsilon:
#             action = np.random.randint(self.env.action_space.n)
#         else:
#             with torch.no_grad():
#                 obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
#                 q_values = self.q_net(obs)
#                 action = torch.argmax(q_values).item()
#         return action

#     def update_q_net(self):
#         if len(self.replay_memory.buffer) < self.params['batch_size']:
#             return
#         # TODO: Update Q-net
#         # HINT: You should draw a batch of random samples from the replay buffer
#         # and train your Q-net with that sampled batch.

#         # predicted_state_value = ???
#         # target_value = ???

#         # Use target network to estimate target Q-values


#         # Update Q-net


#         # # Use target network to estimate target Q-values
#         # Draw a batch of random samples from the replay buffer
#         # batch_size = self.params['batch_size']
#         # state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*self.replay_memory.sample(self.params['batch_size']))

#         # # Assume that next_state_batch is a list of arrays, some of which may be None
#         # valid_next_state_batch = []
#         # for x in next_state_batch:
#         #     if x is not None:
#         #         valid_next_state_batch.append(x)

#         # max_size = np.max([x.shape[0] if x is not None else 0 for x in valid_next_state_batch])
#         # padded_next_state_batch = np.zeros((len(valid_next_state_batch), max_size) + valid_next_state_batch[0].shape[1:], dtype=valid_next_state_batch[0].dtype)
#         # for i, x in enumerate(valid_next_state_batch):
#         #     if x is not None:
#         #         padded_next_state_batch[i, :x.shape[0]] = x
        
#         # # Convert batches to tensors
#         # state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32, device=get_device())
#         # action_batch = torch.tensor(np.array(action_batch), dtype=torch.long, device=get_device())
#         # reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=get_device())
#         # next_state_batch = torch.tensor(np.array(padded_next_state_batch), dtype=torch.float32, device=get_device())
#         # done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32, device=get_device())
        
#         # # Use target network to estimate target Q-values
#         # with torch.no_grad():
#         #     # Compute the target Q-values for the next state
#         #     # Compute the predicted Q-values for the current state and action
#         #     print()
#         # predicted_q_values = self.q_net(state_batch).gather(dim=1, index=action_batch.unsqueeze(1)).squeeze(1)




#         batch = self.replay_memory.sample(self.params['batch_size'])
#         obs_batch, action_batch, reward_batch, next_obs_batch, not_done_mask_batch = zip(*batch)

#         obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(get_device())
#         action_tensor = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(get_device())
#         reward_tensor = torch.tensor(reward_batch, dtype=torch.float32).to(get_device())
#         next_obs_tensor = torch.tensor([obs for obs in next_obs_batch if obs is not None], dtype=torch.float32).to(get_device())
#         not_done_mask_tensor = torch.tensor(not_done_mask_batch, dtype=torch.bool).to(get_device())

#         predicted_state_value = self.q_net(obs_tensor).gather(1, action_tensor)

#         with torch.no_grad():
#             target_value = torch.zeros(self.params['batch_size']).to(get_device())
#             target_value[not_done_mask_tensor] = self.target_net(next_obs_tensor).max(dim=1)[0].detach()
#             target_value = reward_tensor + self.params['gamma'] * target_value



#         #############################
#         # Compute the Q-loss
#         criterion = nn.SmoothL1Loss()
#         q_loss = criterion(predicted_state_value, target_value.unsqueeze(1))
#         self.optimizer.zero_grad()
#         q_loss.backward()
#         self.optimizer.step()

#     def update_target_net(self):
#         if len(self.replay_memory.buffer) < self.params['batch_size']:
#             return
#         q_net_state_dict = self.q_net.state_dict()
#         target_net_state_dict = self.target_net.state_dict()
#         for key in q_net_state_dict:
#             target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (1 - self.params['tau'])*target_net_state_dict[key]
#         self.target_net.load_state_dict(target_net_state_dict)

#     def generate_video(self, max_frame=1000):
#         self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
#         self.epsilon = 0.0
#         obs, _ = self.env.reset()
#         for _ in range(max_frame):
#             action = self.get_action(obs)
#             obs, reward, terminated, truncated, info = self.env.step(action)
#             if terminated or truncated:
#                 break
#         save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


# class ReplayMemory:
#     # TODO: Implement replay buffer
#     # HINT: You can use python data structure deque to construct a replay buffer
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)

#     def push(self, *args):
#         self.buffer.append(tuple(args))

#     def sample(self, n_samples):
#        return random.sample(self.buffer, n_samples)


# class QNet(nn.Module):
#     # TODO: Define Q-net
#     # This is identical to policy network from HW1
#     def __init__(self, input_size, output_size, hidden_dim):
#         super(QNet, self).__init__()
#         self.ff_net = nn.Sequential(
#             nn.Linear(input_size,hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim,output_size),
#             nn.Softmax(dim = -1)
#         )

#     def forward(self, obs):
#         return self.ff_net(obs)



import copy
import pickle
import random
import gymnasium as gym
import torch
from collections import deque, namedtuple
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *
import torch.nn.functional as F
import itertools


# Class for training an RL agent with Actor-Critic
class ACTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = ACAgent(env=self.env, params=self.params)
        self.actor_net = ActorNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.critic_net = CriticNet(input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.actor_optimizer = Adam(params=self.actor_net.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = Adam(params=self.critic_net.parameters(), lr=self.params['critic_lr'])
        self.trajectory = None

    def run_training_loop(self):
        list_ro_reward = list()
        for ro_idx in range(self.params['n_rollout']):
            self.trajectory = self.agent.collect_trajectory(policy=self.actor_net)
            self.update_critic_net()
            self.estimate_advantage()
            self.update_actor_net()
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            # avg_ro_reward = ???
            
            avg_ro_reward = sum([sum(traj) for traj in self.trajectory["reward"]]) / len(self.trajectory["reward"])
            print(f'Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def update_critic_net(self):
        for critic_iter_idx in range(self.params['n_critic_iter']):
            self.update_target_value()
            for critic_epoch_idx in range(self.params['n_critic_epoch']):
                critic_loss = self.estimate_critic_loss_function()
                critic_loss.requires_grad_(True)
                critic_loss.backward()
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

    def update_target_value(self, gamma=0.99):
        # TODO: Update target values
        # HINT: Use definition of target-estimate from equation 7 of teh assignment PDF

        # self.trajectory['state_value'] = ???
        # self.trajectory['target_value'] = ???
        
        # self.trajectory['state_value'] = [self.critic_net(obs).detach() for obs in self.trajectory['obs']]
        # self.trajectory['target_value'] = [reward + gamma * next_value for reward, next_value in zip(self.trajectory['reward'], self.trajectory['state_value'][1:] + [0])]
        # state_tensor = torch.tensor(self.trajectory['obs'], dtype=torch.float32, device=get_device())
        # state_value_tensor = self.critic_net(state_tensor).squeeze().detach().cpu().numpy()
        # self.trajectory['state_value'] = state_value_tensor
        # self.trajectory['target_value'] = [reward + gamma * next_value for reward, next_value in zip(self.trajectory['reward'], self.trajectory['state_value'][1:])]
        
        state_values = []
        for state in self.trajectory['obs']:
            state_tensor = state.to(get_device())
            state_value = self.critic_net(state_tensor).squeeze().detach().cpu().numpy()
            state_values.append(state_value)
        self.trajectory['state_value'] =    np.concatenate(state_values)
        self.trajectory['target_value'] = [reward + gamma * next_value for reward, next_value in zip(self.trajectory['reward'][:-1], self.trajectory['state_value'][1:])]        


    def estimate_advantage(self, gamma=0.99):
        # TODO: Estimate advantage
        # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF

        # self.trajectory['advantage'] = ???
        self.trajectory['advantage'] = [reward + gamma * next_value for reward, next_value in zip(self.trajectory['reward'], self.trajectory['state_value'][1:] + [torch.tensor(0, device=get_device())])]


    def update_actor_net(self):
        actor_loss = self.estimate_actor_loss_function()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def estimate_critic_loss_function(self):
        new_size = max(len(self.trajectory['state_value']), len(self.trajectory['target_value'][0]))
        
        state_value_padded = np.pad(self.trajectory['state_value'], (0, new_size - len(self.trajectory['state_value'])), mode='constant', constant_values=0)
        target_value_padded = np.pad(self.trajectory['target_value'][0], (0, new_size - len(self.trajectory['target_value'][0])), mode='constant', constant_values=0)
        # TODO: Compute critic loss function
        # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.
        # critic_loss = ???
        # critic_loss = torch.nn.MSELoss()(torch.stack(self.trajectory['state_value']), torch.tensor(self.trajectory['target_value'][0], device=get_device()))
        critic_loss = torch.nn.MSELoss()(torch.stack(tuple(torch.tensor(self.trajectory['state_value'],device=get_device()))), torch.tensor(target_value_padded, device=get_device()))
        return critic_loss

    def estimate_actor_loss_function(self):
        actor_loss = list()
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            advantage = apply_discount(self.trajectory['advantage'][t_idx])
            actor_loss.append(-self.trajectory['log_prob'][t_idx] * advantage)
        # TODO: Compute actor loss function
        # actor_loss = ???
        pad_value= 0
        max_size = max(tensor.size(0) for tensor in actor_loss)
        padded_actor_loss = [
            (lambda t: torch.cat((t, torch.full((max_size - t.size(0),), pad_value, dtype=t.dtype, device=t.device)), dim=0))(tensor)
            for tensor in actor_loss

        ]
        actor_loss = torch.mean(torch.stack(padded_actor_loss))
        return actor_loss

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_net(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


# CLass for actor-net
class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(ActorNet, self).__init__()
        # TODO: Define the actor net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        # self.ff_net = ???
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        # TODO: Forward pass of actor net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        # action_index = ???
        # log_prob = ???
        probs = self.ff_net(obs)
        dist = Categorical(probs)
        action_index =  dist.sample()
        log_prob     = dist.log_prob(action_index)
        
        return action_index, log_prob


# CLass for actor-net
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(CriticNet, self).__init__()
        # TODO: Define the critic net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        # self.ff_net = ???
        self.ff_net =  nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        # TODO: Forward pass of critic net
        # HINT: (get state value from the network using the current observation)
        # state_value = ???
        state_value = self.ff_net(obs)
        return state_value


# Class for agent
class ACAgent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
            while True:
                obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
                # Save observation
                trajectory_buffer['obs'].append(obs)
                action_idx, log_prob = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['obs'].append(torch.stack(trajectory_buffer['obs']))
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer






































class DQNTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.q_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.params['init_epsilon']
        self.optimizer = Adam(params=self.q_net.parameters(), lr=self.params['lr'])
        self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

    def run_training_loop(self):
        list_ep_reward = list()
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        for idx_episode in range(self.params['n_episode']):
            ep_len = 0
            while True:
                ep_len += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    self.epsilon = max(self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
                    next_obs = None
                    self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                    list_ep_reward.append(ep_len)
                    print(f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
                    obs, _ = self.env.reset()
                    break
                self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                obs = copy.deepcopy(next_obs)
                self.update_q_net()
                self.update_target_net()
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ep_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def get_action(self, obs):
        # TODO: Implement the epsilon-greedy behavior
        # HINT: The agent will will choose action based on maximum Q-value with
        # '1-ε' probability, and a random action with 'ε' probability.
        # action = ???
        if torch.rand(1).item() > self.epsilon:
            obs_tensor = torch.FloatTensor(obs).to(get_device())
            with torch.no_grad():
                action = self.q_net(obs_tensor).argmax().item()
        else:
            action = self.env.action_space.sample()
        return action

    def update_q_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        # TODO: Update Q-net
        # HINT: You should draw a batch of random samples from the replay buffer
        # and train your Q-net with that sampled batch.

        # predicted_state_value = ???
        # target_value = ???
        batch = self.replay_memory.sample(self.params['batch_size'])
        obs_batch, action_batch, reward_batch, next_obs_batch, not_done_mask_batch = zip(*batch)

        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(get_device())
        action_tensor = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(get_device())
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32).to(get_device())
        next_obs_tensor = torch.tensor([obs for obs in next_obs_batch if obs is not None], dtype=torch.float32).to(get_device())
        not_done_mask_tensor = torch.tensor(not_done_mask_batch, dtype=torch.bool).to(get_device())

        predicted_state_value = self.q_net(obs_tensor).gather(1, action_tensor)

        with torch.no_grad():
            target_value = torch.zeros(self.params['batch_size']).to(get_device())
            target_value[not_done_mask_tensor] = self.target_net(next_obs_tensor).max(dim=1)[0].detach()
            target_value = reward_tensor + self.params['gamma'] * target_value


        criterion = nn.SmoothL1Loss()
        q_loss = criterion(predicted_state_value, target_value.unsqueeze(1))
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        q_net_state_dict = self.q_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (1 - self.params['tau'])*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        self.epsilon = 0.0
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


class ReplayMemory:

    def __init__(self, capacity):

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)


    def push(self, *args):

        self.buffer.append(args)

    def sample(self, n_samples):
        if len(self.buffer) < n_samples:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, n_samples)
    
    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    # TODO: Define Q-net
    # This is identical to policy network from HW1
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNet, self).__init__()
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.ff_net(obs)
