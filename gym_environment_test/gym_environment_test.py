# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:18:20 2017

@author: Kuanho

Environment description:
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. 
Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. 
If lander moves away from landing pad it loses reward back. 
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
Each leg ground contact is +10. Firing main engine is -0.3 points each frame. 
Solved is 200 points. Landing outside landing pad is possible. 
Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 
Action is two real values vector from -1 to +1. 
First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power. 
Engine can't work with less than 50% power. 
Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
"""


import gym
import numpy as np
env = gym.make('Pendulum-v0')

for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        #use 'env.render()'to see animation
        env.render()
        action =np.array([1,0,0])
        observation, reward, done, info = env.step(action)
        print(observation)
        print(action)
        print(reward)
        while (input("Press Enter to continue...")): print('wait')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.render(close=True)