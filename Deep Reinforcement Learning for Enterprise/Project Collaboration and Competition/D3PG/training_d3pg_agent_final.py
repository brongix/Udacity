# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import torch
import numpy as np
from collections import deque
import time

from d3pg_agent_final import Agent


def d3pg(env, dic = {'n_episodes': 1000,
       'max_t': 300,
       'actor_fc1_units': 400,
       'actor_fc2_units': 300,
       'critic_fc1_units': 400,
       'critic_fc2_units': 300,
       'buffer_size': int(1e6),  # replay buffer size
       'batch_size': 128,        # minibatch size
       'gamma': 0.99,            # discount factor
       'tau': 1e-3,              # for soft update of target parameters
       'lr_actor': 1e-4,         # learning rate of the actor 
       'lr_critic': 1e-4,        # learning rate of the critic
       'weight_decay': 0,        # L2 weight decay
       'update_every': 1,
       'epsilon': 0.3,
       'file': 'chpt',
       }, SuccessStop=True):

    
    #unfolding dic
    n_episodes = dic['n_episodes']
    max_t = dic['max_t']
    file = dic['file']
    
    
 
  
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # number of actions
    action_size = brain.vector_action_space_size
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    
    # number of agents
    #num_agents = len(env_info.agents)
    
    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    # initialise the agent
    agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, hyper_dic=dic)
    
    scores_list = []                         # list containing scores from each episode
    times = []                        # list containing times for each episode
    scores_window = deque(maxlen=100)  # last 100 scores
   
    start_time = time.time()            # initialise time  for the whole training process 
    split_time = time.time()              # initialise 100 episodes time split
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment in training mode
        states = env_info.vector_observations            # get the current state
        scores = np.zeros(num_agents)
        initial_time = time.time()                          # initialise episode time split
                                   
        for t in range(max_t):
            actions = agent.act(states, add_noise=True, episode=i_episode)
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break 
        scores_window.append(np.max(scores))                    # save most recent score
        scores_list.append(np.max(scores))             # save most recent average score
        times.append(time.time() - initial_time) # save most recent time
        
        print('\rEpisode {}\tAverage Score: {:.2f}\tTime Elapsed: {:.2f}'.format(i_episode, np.mean(scores_window), time.time() - start_time), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime Split: {:.2f}\t\t\t '.format(i_episode, np.mean(scores_window), time.time() - split_time))
            split_time = time.time()
        if np.mean(scores_window)>=0.5 and SuccessStop:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime Elapsed: {:.2f}'.format(i_episode-100, np.mean(scores_window), time.time() - start_time))
            break
    if np.mean(scores_window)>=0.5:
        torch.save(agent.actor_local.state_dict(), file+'_actor.pth')
        torch.save(agent.critic_local.state_dict(), file+'_critic.pth')
    return scores_list, times
