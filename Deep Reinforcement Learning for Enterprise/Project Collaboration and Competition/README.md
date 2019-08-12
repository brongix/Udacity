[//]: # (Image References)

[image1]: https://github.com/brongix/Udacity/blob/master/Deep%20Reinforcement%20Learning%20for%20Enterprise/Project%20Collaboration%20and%20Competition/Trained%20MADDPG%20Agents.gif "Header"



#### Currently solved with a Tuned D3PG model in *835 episodes*.


# 

# Project 3: Collaboration and Competition

This is the third and last project in a series of projects linked to the *Deep Reinforcement Learning for Enterprise Nanodegree Program* at *Udacity*.

![Header][image1]

## Project Details

In the Unity's [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, I trained two agents to control rackets in order to bounce a ball over a net. 


If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.


The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

<br/><br/>

## Getting Started

Below are the generic instructions needed to set the environment up in the various operating systems: 
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file.  


I have set up a local virtual environment - `unity` - in a Mac OS system as suggested in the instructions in the [DRLND Repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). I have kept the environment light only installing further [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [PyTorch](https://pytorch.org/), [NumPy](http://www.numpy.org/) and [Pandas](https://pandas.pydata.org/). I include `unity.yml` in this repo so `unity` can be replicated.

<br/><br/>


## Instructions

This repo is currently organised into four folders where we used different versions of D3PG and MADDPG models,


	- D3PG
	- D3PG with NoisyNets and PER
	- MADDPG
	- MADDPG with Collaboration Level and Energy Expenditure


<br/><br/>
In the *D3PG* folder, we have the *D3PG_analysis* notebook where we analysed and tuned the D3PG model learning performance for the Tennis environment with both agents associated to the same policy.


In the *D3PG with NoisyNets and PER* folder, we have the *D3PG_nn_per_analysis* notebook where we enhanced the D3PG model incorporating Noisy Networks and Prioritised Experince Replay. We analysed the impact of each new component in the learning performance of the model.


In the *MADDPG* folder, we have the *MADDPG_analysis* notebook where we implemented the MADDPG model. We analysed its learning performance for the Tennis environmet with each agent learning a seperate policy.


In the *MADDPG with Collaboration Level and Energy Expenditure* folder, we have the *MADDPG_cl_ee_analysis* notebook where we enhanced the MADDPG model with an explicit collaboration control level between the agents and some energy expenditure constraint in order to improve the agent movement without the ball. We analysed the impact in the learning performance and the quality of the trained agent movement without the ball.

<br/><br/>
Each notebook contains description of the code being executed and I present a summary of my analysis and results in the `report.pdf` file also included in this repo.

For more detailed videos on the different stages of performance of the agents in the environment please check out my [YouTube channel](https://www.youtube.com/channel/UCR0dwjdcbswHvQvnfC8IW-g/videos?view_as=subscriber).
