# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from unityagents import UnityEnvironment

import torch
import numpy as np
from collections import deque
import time

from dqn_agent import Agent


def dqn(env, dic = {'n_episodes': 2000,
       'max_t': 1000,
       'eps_start': 1.0,
       'eps_end': 0.01,
       'eps_decay':0.995,
       'fc1_units':64,
       'fc2_units':64,
       'buffer_size': int(1e5),
       'batch_size': 64,
       'gamma':0.99,
       'tau': 1e-3,
       'lr': 1e-4,
       'update_every': 4
      }):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    #unfolding dic
    n_episodes = dic['n_episodes']
    max_t = dic['max_t']
    eps_start = dic['eps_start']
    eps_end = dic['eps_end']
    eps_decay = dic['eps_decay']
    fc1_units = dic['fc1_units']
    fc2_units = dic['fc2_units']
    hyper_dic={}
    hyper_dic['buffer_size'] = dic['buffer_size']
    hyper_dic['batch_size'] = dic['batch_size']
    hyper_dic['gamma'] = dic['gamma']
    hyper_dic['tau'] = dic['tau']
    hyper_dic['lr'] = dic['lr']
    hyper_dic['update_every'] = dic['update_every']
  
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # number of actions
    action_size = brain.vector_action_space_size
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    # initialise the agent
    agent = Agent(state_size=state_size, action_size=action_size, fc1_units=fc1_units, fc2_units=fc2_units, seed=0, hyper_dic=hyper_dic)
    
    scores = []                         # list containing scores from each episode
    times = []                        # list containing times for each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialise epsilon
    start_time = time.time()            # initialise time  for the whole training process 
    split_time = time.time()              # initialise 100 episodes time split
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment in training mode
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        initial_time = time.time()                          # initialise episode time split
                                   
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)             # save most recent score
        times.append(time.time() - initial_time) # save most recent time
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tTime Elapsed: {:.2f}'.format(i_episode, np.mean(scores_window), time.time() - start_time), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime Split: {:.2f}\t\t\t '.format(i_episode, np.mean(scores_window), time.time() - split_time))
            split_time = time.time()
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime Elapsed: {:.2f}'.format(i_episode-100, np.mean(scores_window), time.time() - start_time))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_final.pth')
            break
    return scores, times
