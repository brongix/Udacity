# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import torch
import numpy as np
from collections import deque
import time

from ddpg_agent_final import Agent


def ddpg(env, dic = {'n_episodes': 200,
       'max_t': 500,
       'actor_fc1_units': 400,
       'actor_fc2_units': 300,
       'critic_fc1_units': 400,
       'critic_fc2_units': 300,
       'buffer_size': int(1e5),  # replay buffer size
       'batch_size': 128,        # minibatch size
       'gamma': 0.99,            # discount factor
       'tau': 1e-3,              # for soft update of target parameters
       'lr_actor': 1e-4,         # learning rate of the actor 
       'lr_critic': 1e-3,        # learning rate of the critic
       'weight_decay': 0,        # L2 weight decay
       'update_every': 4,
       'noise_theta': 0.15,
       'noise_sigma': 0.2,
       'file': 'checkpoint'}):

    
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
    
    # number of agents
    #num_agents = len(env_info.agents)
    
    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    # initialise the agent
    agent = Agent(state_size=state_size, action_size=action_size, hyper_dic=dic)
    
    scores = []                         # list containing scores from each episode
    times = []                        # list containing times for each episode
    scores_window = deque(maxlen=100)  # last 100 scores
   
    start_time = time.time()            # initialise time  for the whole training process 
    split_time = time.time()              # initialise 100 episodes time split
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment in training mode
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        initial_time = time.time()                          # initialise episode time split
                                   
        for t in range(max_t):
            action = agent.act(state, add_noise=True, episode=i_episode)
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
        
        print('\rEpisode {}\tAverage Score: {:.2f}\tTime Elapsed: {:.2f}'.format(i_episode, np.mean(scores_window), time.time() - start_time), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime Split: {:.2f}\t\t\t '.format(i_episode, np.mean(scores_window), time.time() - split_time))
            split_time = time.time()
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime Elapsed: {:.2f}'.format(i_episode-100, np.mean(scores_window), time.time() - start_time))
            torch.save(agent.actor_local.state_dict(), file+'_actor.pth')
            torch.save(agent.critic_local.state_dict(), file+'_critic.pth')
            break
        
        
    return scores, times
