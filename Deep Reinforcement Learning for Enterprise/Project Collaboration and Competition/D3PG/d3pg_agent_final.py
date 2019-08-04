import numpy as np
import random
#import copy
from collections import namedtuple, deque

from model_d3pg_final import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents, state_size, action_size, seed=0, hyper_dic = {'actor_fc1_units': 400,
                                                             'actor_fc2_units': 300,
                                                             'critic_fc1_units': 400,
                                                             'critic_fc2_units': 300,
                                                             'buffer_size': int(1e6),  # replay buffer size
                                                             'batch_size': 256,        # minibatch size
                                                             'gamma': 0.99,            # discount factor
                                                             'tau': 1e-3,              # for soft update of target parameters
                                                             'lr_actor': 1e-4,         # learning rate of the actor 
                                                             'lr_critic': 1e-4,        # learning rate of the critic
                                                             'weight_decay': 0,        # L2 weight decay
                                                             'update_every': 1,
                                                             'epsilon': 0.3,
                                                             
                                                             }):
    

        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        self.BATCH_SIZE = hyper_dic['batch_size']        # minibatch size
        self.GAMMA = hyper_dic['gamma']           # discount factor
        self.TAU = hyper_dic['tau']            # for soft update of target parameters
        self.epsilon = hyper_dic['epsilon']
        self.UPDATE_EVERY = hyper_dic['update_every']
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        #self.seed = random.seed(seed)
        self.t_step = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed, fc1_units=hyper_dic['actor_fc1_units'], fc2_units=hyper_dic['actor_fc2_units']).to(device)
        self.actor_target = Actor(state_size, action_size, seed, fc1_units=hyper_dic['actor_fc1_units'], fc2_units=hyper_dic['actor_fc2_units']).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyper_dic['lr_critic'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed, fcs1_units=hyper_dic['critic_fc1_units'], fc2_units=hyper_dic['critic_fc2_units']).to(device)
        self.critic_target = Critic(state_size, action_size, seed, fcs1_units=hyper_dic['critic_fc1_units'], fc2_units=hyper_dic['critic_fc2_units']).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyper_dic['lr_critic'], weight_decay=hyper_dic['weight_decay'])

        
        # Replay memory
        self.memory = ReplayBuffer(action_size, hyper_dic['buffer_size'], hyper_dic['batch_size'])
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)
       


    def act(self, states, add_noise=True, episode=None):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
            #print(actions)
        self.actor_local.train()
        if add_noise:
            noise = (.5)**(episode/500)*self.epsilon*np.random.randn(self.num_agents, self.action_size)
            #print(noise)
            actions += noise
            actions = np.clip(actions, -1, 1)
            #print(actions)
        return actions

    

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)