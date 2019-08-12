import numpy as np
import random
#import copy
from collections import namedtuple, deque

from model_maddpg import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from prioritized_memory import Memory 

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
                                                             'epsilon_halflife': 500,
                                                             
                                                             }, PER=True):
    

        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
                  
        self.epsilon = hyper_dic['epsilon']
        self.halflife = hyper_dic['epsilon_halflife']
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.t_step = 0
        self.hyper_dic = hyper_dic
        self.PER = PER

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed, fc1_units=hyper_dic['actor_fc1_units'], fc2_units=hyper_dic['actor_fc2_units']).to(device)
        self.actor_target = Actor(state_size, action_size, seed, fc1_units=hyper_dic['actor_fc1_units'], fc2_units=hyper_dic['actor_fc2_units']).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyper_dic['lr_critic'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, num_agents, seed, fcs1_units=hyper_dic['critic_fc1_units'], fc2_units=hyper_dic['critic_fc2_units']).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, seed, fcs1_units=hyper_dic['critic_fc1_units'], fc2_units=hyper_dic['critic_fc2_units']).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyper_dic['lr_critic'], weight_decay=hyper_dic['weight_decay'])

   


    def act(self, states, add_noise=True, episode=None, NoiseNet=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
            
        self.actor_local.train()
        if add_noise:
            if NoiseNet:
                NN = Actor(self.state_size, self.action_size, self.seed, fc1_units=self.hyper_dic['actor_fc1_units'], fc2_units=self.hyper_dic['actor_fc2_units']).to(device)
                actor_state_dict = self.actor_local.state_dict()
                noisy_state_dict = {}
                for key in actor_state_dict:
                    noisy_state_dict[key] = actor_state_dict[key] + torch.from_numpy((.5)**(episode/self.halflife)*self.epsilon*np.random.normal(size=actor_state_dict[key].shape)).float()
                NN.load_state_dict(noisy_state_dict)
                NN.eval()
                with torch.no_grad():
                    actions = NN(states).cpu().data.numpy()
            else:    
                noise = (.5)**(episode/self.halflife)*self.epsilon*np.random.normal(size=actions.shape)
                actions += noise
            actions = np.clip(actions, -1, 1)
            
        return actions

    

 