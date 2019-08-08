[//]: # (Image References)

[image1]: https://github.com/brongix/Udacity/blob/master/Deep%20Reinforcement%20Learning%20for%20Enterprise/Project%20Continuous%20Control/Header.gif "Header"
[image2]: https://github.com/brongix/Udacity/blob/master/Deep%20Reinforcement%20Learning%20for%20Enterprise/Project%20Continuous%20Control/one_agent.gif "One Agent"
[image3]: https://github.com/brongix/Udacity/blob/master/Deep%20Reinforcement%20Learning%20for%20Enterprise/Project%20Continuous%20Control/twenty_agents.gif "Twenty Agents"


#### Currently solved with a Tuned D3PG model on average in less than *25 episodes*.
#### Shortest learning session to date *19 episodes*.

# 

# Project 2: Continuous Control

This is the second project in a series of projects linked to the *Deep Reinforcement Learning for Enterprise Nanodegree Program* at *Udacity*.

![Header][image1]

## Project Details

In the Unity's Reacher environment, I trained double-jointed arms to move to target locations.



A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

<br/><br/>

## Getting Started

Below are the generic instructions needed to set the environment up in the various operating systems: 
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 
 


I have set up a local virtual environment - `unity` - in a Mac OS system as suggested in the instructions in the [DRLND Repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). I have kept the environment light only installing further [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [PyTorch](https://pytorch.org/), [NumPy](http://www.numpy.org/) and [Pandas](https://pandas.pydata.org/). I include `unity.yml` in this repo so `unity` can be replicated.

<br/><br/>


## Instructions

This repo is currently organised into two folders where we used different versions of Unity's Reachers environment,


	- One Agent
	- Twenty Agents


<br/><br/>
In the *One Agent* folder, we have the *DDPG_analysis* notebook where I tested and tuned the DDPG model learning performance in a Reacher environment with *one agent*.

![One Agent][image2]

In this environment in order to solve it, the agent had to get an average score of +30 over 100 consecutive episodes.
 
<br/><br/>

In the *Twenty Agents* folder, we have the *D3PG_analysis* notebook where I tested and enhanced the D3PG model learning performance in a Reacher environment with *twenty agents*. 

![Twenty Agents][image3]

To take into account the presence of many agents, we only declare the environment solved when the average of the agents score is above +30 over 100 consecutive episodes.


<br/><br/>


Each notebook contains description of the code being executed and I present a summary of my analysis and results in the `report.pdf` file also included in this repo.

For more detailed videos on the different stages of performance of the agents in the different environments please check out my [YouTube channel](https://www.youtube.com/channel/UCR0dwjdcbswHvQvnfC8IW-g/videos?view_as=subscriber).
