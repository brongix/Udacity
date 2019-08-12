import numpy as np
import random
#import copy
from collections import namedtuple, deque

from maddpg_agent_cl_ee import Agent
from model_maddpg import Actor

import torch
import torch.nn.functional as F
import torch.optim as optim

from prioritized_memory import Memory 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Team():
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
                                                             'collaboration_level': 1,
                                                             'energy_expenditure': 0,
                                                             }, PER=True):
    

        """Initialize a Group object.
        
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
        self.halflife = hyper_dic['epsilon_halflife']
        self.UPDATE_EVERY = hyper_dic['update_every']
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.t_step = 0
        self.hyper_dic = hyper_dic
        self.PER = PER
        self.collaboration_level = hyper_dic['collaboration_level']
        self.energy_expenditure = hyper_dic['energy_expenditure']

        self.maddpg_agents = [Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, hyper_dic=hyper_dic, PER=PER) for _ in range(num_agents)]
                             

        
        # Replay memory
        if self.PER:
            self.memory = Memory(hyper_dic['buffer_size'])
            
        else:
            self.memory = ReplayBuffer(action_size, hyper_dic['buffer_size'], hyper_dic['batch_size'])
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if self.PER:
            next_states = torch.from_numpy(np.array(next_states).reshape(1,-1)).float().to(device)
            states = torch.from_numpy(np.array(states).reshape(1,-1)).float().to(device)
            actions = torch.from_numpy(np.array(actions).reshape(1,-1)).float().to(device)
            rewards = torch.from_numpy(np.array(rewards).reshape(1,-1)).float().to(device)
            dones = torch.from_numpy(np.array(dones).reshape(1,-1).astype(np.uint8)).float().to(device)
             
            actions_next = torch.cat([agent.actor_target(next_states[:, i*self.state_size : (i+1)*self.state_size]) for i, agent in enumerate(self.maddpg_agents)], dim=1).float().to(device)
    
            with torch.no_grad():
                Q_targets_next = torch.cat([agent.critic_target(next_states, actions_next) for agent in self.maddpg_agents], dim=1).float().to(device)
                # Compute Q targets for current states (y_i)
            Q_targets = (1-self.collaboration_level) * rewards + self.collaboration_level * torch.sum(rewards) * torch.ones(rewards.shape)  + (self.GAMMA * Q_targets_next * (1 - dones))
            
                # Compute critic loss
            with torch.no_grad():
                Q_expected = torch.cat([agent.critic_local(states, actions) for agent in self.maddpg_agents], dim=1).float().to(device)
            
            error = torch.max(torch.abs(Q_expected - Q_targets), 1)[0].data.numpy()            
        
        if self.PER:
            self.memory.add(float(error), (states, actions, rewards, next_states, dones))               
        else:    
            self.memory.add(np.array(states).reshape(1,-1), np.array(actions).reshape(1,-1), rewards, np.array(next_states).reshape(1, -1), dones)

        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.PER:
                current_memory = self.memory.tree.n_entries    
            else:
                current_memory = len(self.memory)
            if current_memory > self.BATCH_SIZE:
                if self.PER:
                    experiences = self.memory.sample(self.BATCH_SIZE)
                else:
                    experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)
       


    def act(self, states, add_noise=True, episode=None, NoiseNet=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions=[]
        for i, agent in enumerate(self.maddpg_agents):
            agent.actor_local.eval()
            
            if add_noise:
                
                if NoiseNet:
                    NN = Actor(self.state_size, self.action_size, self.seed, fc1_units=self.hyper_dic['actor_fc1_units'], fc2_units=self.hyper_dic['actor_fc2_units']).to(device)
                    actor_state_dict = agent.actor_local.state_dict()
                    noisy_state_dict = {}
                    for key in actor_state_dict:
                        noisy_state_dict[key] = actor_state_dict[key] + torch.from_numpy((.5)**(episode/self.halflife)*self.epsilon*np.random.normal(size=actor_state_dict[key].shape)).float()
                    NN.load_state_dict(noisy_state_dict)
                    NN.eval()
                    with torch.no_grad():
                        actions.append(np.clip(NN(states[i]).cpu().data.numpy(), -1, 1))
                else:
                    agent.actor_local.eval()
                    with torch.no_grad():
                        clean_action = agent.actor_local(states[i]).cpu().data.numpy()      
                    noise = (.5)**(episode/self.halflife)*self.epsilon*np.random.normal(size=clean_action.shape)
                    actions.append(np.clip(clean_action + noise, -1, 1))
            else:
                agent.actor_local.eval()
                with torch.no_grad():
                    clean_action = agent.actor_local(states[i]).cpu().data.numpy()
                actions.append(np.clip(clean_action, -1, 1))  
            
            agent.actor_local.train()
            
        actions = np.array(actions)

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
        if self.PER:
            mini_batch, idxs_i, is_weights_i = experiences
           
            #mini_batch = np.array(mini_batch).transpose()
            states=[]
            actions=[]
            rewards=[]
            next_states=[]
            dones=[]
            idxs=[]
            is_weights=[]
            for i, e in enumerate(mini_batch):
                
                if type(e) != int:
                    idxs.append(idxs_i[i])
                    is_weights.append(is_weights_i[i])
                    states.append(e[0])
                    actions.append(e[1])
                    rewards.append(e[2])
                    next_states.append(e[3])
                    dones.append(e[4])
            
            
            states = torch.from_numpy(np.vstack(states)).float().to(device)
            actions = torch.from_numpy(np.vstack(actions)).float().to(device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
            next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
            dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
            
            
        else:    
            states, actions, rewards, next_states, dones = experiences
            

        # ---------------------------- update critic ---------------------------- #
        
        # Get predicted next-state actions and Q values from target models
        actions_next = torch.cat([agent.actor_target(next_states[:, i*self.state_size : (i+1)*self.state_size]) for i, agent in enumerate(self.maddpg_agents)], dim=1)
        
        with torch.no_grad():
            Q_targets_next = torch.cat([agent.critic_target(next_states, actions_next) for agent in self.maddpg_agents], dim=1)
        # Compute Q targets for current states (y_i)
        Q_targets = (1-self.collaboration_level) * rewards + self.collaboration_level * torch.sum(rewards) * torch.ones(rewards.shape) + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = torch.cat([agent.critic_local(states, actions) for agent in self.maddpg_agents], dim=1)
        
 
            

        if self.PER:
            
            with torch.no_grad():
                Q_expected = torch.cat([agent.critic_local(states, actions) for agent in self.maddpg_agents], dim=1).float().to(device)
            
            errors = torch.max(torch.abs(Q_expected - Q_targets), 1)[0].data.numpy() 

            #with torch.no_grad():
                #errors = torch.abs(Q_targets.detach() - Q_expected).max().data.numpy()
                #errors = np.ones(len(idxs))
        
            # update priority
            for i in range(len(idxs)):
                idx = idxs[i]
                self.memory.update(idx, errors[i])
            Q_expected = torch.cat([agent.critic_local(states, actions) for agent in self.maddpg_agents], dim=1)
            critic_loss = (torch.FloatTensor(is_weights) * F.mse_loss(Q_expected, Q_targets.detach())).mean()
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        for agent in self.maddpg_agents:
            agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        for agent in self.maddpg_agents:
            agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        for j, agent in enumerate(self.maddpg_agents):
            actions_pred = [a.actor_local(states[:, i*self.state_size : (i+1)*self.state_size]) if a==agent else a.actor_local(states[:, i*self.state_size : (i+1)*self.state_size]).detach() for i, a in enumerate(self.maddpg_agents)]
            actions_pred = torch.cat(actions_pred, dim=1)
            actor_loss = -agent.critic_local(states, actions_pred).mean() + self.energy_expenditure * torch.sum(torch.abs(agent.actor_local(states[:, j*self.state_size : (j+1)*self.state_size])))
        # Minimize the loss
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        for agent in self.maddpg_agents:
            self.soft_update(agent.critic_local, agent.critic_target, self.TAU)
            self.soft_update(agent.actor_local, agent.actor_target, self.TAU)                     

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