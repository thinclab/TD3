#!/usr/bin/env python3

"""
This script provides the RD3 Reinforcement Learning algorithm from the paper
"Controlling estimation error in reinforcement learning via Reinforced Operation"
"""

import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize object with everything from parent class
        super(Actor, self).__init__()

        # Define the maximum continuous value for an action
        self.max_action = max_action

        # Define the structure of the actor network
        # state dimension input -> 512 features, 512 features -> 512 features, 512 features -> action dimension output
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        # (JK) Decide on which implementation to use
        # # Pass the input through the actor network and return the output
        # return self.actor(x)

        # Pass input through network, restrict between [-1,1], then multiply by max_action to transform output
        return self.max_action * torch.tanh(self.actor(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Define the structure of the first critic network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Define the structure of the second critic network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.q3 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.q4 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.q5 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state, action):
        # Concatenate the state and action inputs into a single tensor
        sa = torch.cat([state, action], 1)

        # Pass the converted input through all of the critic networks and return the values
        return self.q1(sa), self.q2(sa), self.q3(sa), self.q4(sa), self.q5(sa)

    def Q1(self, state, action):
        # Concatenate the state and action inputs into a single tensor
        sa = torch.cat([state, action], 1)

        # Pass the converted input through the first critic network and return the value
        return self.q1(sa)


class RD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        expl_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        # Define NN for actor and copy it for target network, then define optimizer
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Define NN for critic and copy it for target network, then define optimizer (contains all critics)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Define global variables
        self.action_dim = action_dim  # Dimension size of action
        self.max_action = max_action  # Maximum continuous action value
        self.discount = discount  # Multiplication coefficient in update to discount future rewards
        self.tau = tau  # Ratio of current network to update target network with (stability)
        self.expl_noise = expl_noise  # Amount of noise to add to action selection during exploration
        self.policy_noise = policy_noise  # Amount of noise to add to action selection
        self.noise_clip = noise_clip  # Maximum noise to add
        self.policy_freq = policy_freq  # How often to update policy
        self.total_it = 0  # Tracking variable for number of iterations
        self.prev_rewards = []  # Tracking variable for previous rewards

    def select_action(self, state, add_noise=False):
        # Convert the state to a row vector tensor and then add it to the GPU
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)

        # Pass the state through the actor network to get the action
        action = self.actor(state)

        # If the "add_noise" flag is set to True
        if add_noise:
            # Get random noise based on Gaussian with 0 mean and "expl_noise" variance with action size
            noise = (torch.randn_like(action) * self.expl_noise).clamp(-self.noise_clip, self.noise_clip)

            # Add the noise to the action
            action = action + noise

        # Clamp final action to valid bounds
        action = action.clamp(-self.max_action, self.max_action)

        return action.cpu().detach().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # Increment the iteration tracking variable
        self.total_it += 1

        # Sample the replay buffer, this returns tensors for each variable
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Do the following with gradient calculation disabled
        with torch.no_grad():
            # Get random noise based on Gaussian with 0 mean and "policy_noise" variance with action size
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            # Add the noise to all of the sampled "next states", then get the target policy's action
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Get the Q values from the critic target networks
            target_Q1, target_Q2, target_Q3, target_Q4, target_Q5 = self.critic_target(next_state, next_action)

            # Concatenate the tensors along a new dimension, then remove extra dimension, necessary for sorting
            # (batch_size, 1) -> (batch_size, 5, 1) -> (batch_size, 5)
            target_Qs = torch.stack([target_Q1, target_Q2, target_Q3, target_Q4, target_Q5], dim=1).squeeze(-1)

            # Sort the tensors in ascending order
            Qs_sorted, _ = torch.sort(target_Qs, dim=1)

            # Calculate the quasi-median by averaging the 2nd and 3rd smallest Q-values, then restore 2D shape
            # (batch_size, 5) -> (batch_size,) -> (batch_size, 1)
            q_RO = (0.5 * (Qs_sorted[:, 1] + Qs_sorted[:, 2])).unsqueeze(1)

            # Get the target Q value using the reward and "reinforced" Q value
            target_Q = reward + not_done * self.discount * q_RO

        # Get the current Q estimates from both critics
        current_Q1, current_Q2, current_Q3, current_Q4, current_Q5 = self.critic(state, action)

        # Compute the mean squared error loss for both critics, takes tensors and computes scalar
        critic_loss = (
            F.mse_loss(current_Q1, target_Q)
            + F.mse_loss(current_Q2, target_Q)
            + F.mse_loss(current_Q3, target_Q)
            + F.mse_loss(current_Q4, target_Q)
            + F.mse_loss(current_Q5, target_Q)
        )

        # Optimize the critics
        self.critic_optimizer.zero_grad()  # Clear old gradients
        critic_loss.backward()  # Backpropagate the loss
        self.critic_optimizer.step()  # Update the parameters

        # Update the policy every few steps
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss, negated to find maximum (optimizers find minimum), take the mean() to get a scalar
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()  # Clear old gradient
            actor_loss.backward()  # Backpropagate the loss
            self.actor_optimizer.step()  # Update the parameters

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        np.save(filename + "_total_it.npy", self.total_it)
        np.save(filename + "_prev_rewards.npy", self.prev_rewards)

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = int(np.load(filename + "_total_it.npy"))
        self.prev_rewards = np.load(filename + "_prev_rewards.npy").tolist()

    def evaluate_policy(self, reward):
        # Add the most recent reward to the list of previous rewards
        self.prev_rewards.append(reward)

        # If there are more than 20 previous rewards
        if len(self.prev_rewards) >= 20:
            # Remove the oldest reward
            self.prev_rewards.pop(0)

            # Calculate the average reward over the last 20 rewards
            avg_reward = sum(self.prev_rewards) / 20

        # Otherwise, the reward average is 0
        else:
            avg_reward = 0

        return avg_reward
