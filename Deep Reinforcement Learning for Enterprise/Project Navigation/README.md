[//]: # (Image References)

[image3]: https://github.com/brongix/Udacity/blob/master/Deep%20Reinforcement%20Learning%20for%20Enterprise/Project%20Navigation/Agent%20Training.gif "Trained Agent"

# Project 1: Navigation

This is the first project in a series of projects linked to the *Deep Reinforcement Learning for Enterprise Nanodegree Program* at *Udacity*.



## Project Details

For this project, I trained an agent to navigate and collect bananas in a large squared world.  

![Trained Agent][image3]

<iframe width="560" height="315" src="https://www.youtube.com/embed/VlgFuyv_-9c?controls=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

Below are the generic instructions needed to set the environment up in the various operating systems: 
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 


I have set up a local virtual environment - `bananaenv` - in a Mac OS system as suggested in the instructions in the [DRLND Repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). I have kept the environment light only installing further [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [PyTorch](https://pytorch.org/), [NumPy](http://www.numpy.org/) and [Pandas](https://pandas.pydata.org/). I include `bananaenv.yml` and `bananaenv.txt` in this repo so `bananaenv` can be replicated.




### Instructions

This repo is currently organised into four notebooks:

0. Navigation Deep Q-Network Agent Baseline
1. Baseline DQN Model Analysis
2. Hyperparameters Tuning
3. Double DQN Agent Analysis

<br/><br/>
In the first notebook - *Navigation Deep Q-Network Agent Baseline* - we introduce the DQN Agent used to solve the `LunarLander-v2` OpenAI environment in the nanodegree class and adapt it to solve the `Banana Collector` Unity environment.

The notebook is devided in three parts:

1. System, environement, agent get initialised and explored 


<iframe width="560" height="315" src="https://www.youtube.com/embed/KaF6uVCsZ0Y?controls=0" ; frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

 <br/><br/> 
  

2. DQN Agent gets trained until it solves the environment

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/4oqYECDkCBc?controls=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br/><br/>

3. And finally we get to see how a trained DQN Agent performs in the environment

<iframe width="560" height="315" src="https://www.youtube.com/embed/VlgFuyv_-9c?controls=0&amp;start=30" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br/><br/>
<br/><br/>

The rest of the notebooks build on top of this model aiming at improving learning performance of the agent.

In *notebook 1*, we first analyse the performance distribution of the model by running multiple learning sessions.

Next we try to improve the model by tuning its hyperparameters in *notebook 2*.

And finally we start to change aspects of the model to incorporate more recent breakthroughs - in *notebook 3* we analyse the performance of a Double DQN agent.

Each notebook contains detailed description of the code being executed step-by-step and we present a summary of our analysis and results in the `report.pdf` file also included in this repo.
