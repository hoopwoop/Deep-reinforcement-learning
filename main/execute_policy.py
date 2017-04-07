# -*- coding: utf-8 -*-
"""
author: kuanho
execute learned policy
"""

import tensorflow as tf
import numpy as np
import os
import gym

SUMMARY_DIR = os.path.join(os.getcwd(), 'results', 'tf_ddpg')
ENV_NAME = 'LunarLanderContinuous-v2'
RANDOM_SEED = 1234
MAX_EPISODES = 100
MAX_EP_STEPS = 1000

env = gym.make(ENV_NAME)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high


def actor(states, sess, ini=False):
    layer1_size = 400
    layer2_size = 300       
    states = np.float32(states)
    aW1 = tf.Variable(tf.random_uniform([state_dim, layer1_size],-1/np.sqrt(state_dim),1/np.sqrt(state_dim)), name="aW1")
    aB1 = tf.Variable(tf.random_uniform([layer1_size],-1/np.sqrt(state_dim),1/np.sqrt(state_dim)), name="aB1")
    aW2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size],-1/np.sqrt(layer1_size),1/np.sqrt(layer1_size)), name="aW2")
    aB2 = tf.Variable(tf.random_uniform([layer2_size],-1/np.sqrt(layer1_size),1/np.sqrt(layer1_size)), name="aB2")
    aW3 = tf.Variable(tf.random_uniform([layer2_size, action_dim],-3e-3,3e-3), name="aW3")
    aB3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3), name="aB3")
    
    if ini: 
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({'aW1':aW1, 'aB1':aB1, 'aW2':aW2, 'aB2':aB2, 'aW3':aW3, 'aB3':aB3})
        saver.restore(sess, os.path.join(SUMMARY_DIR, 'model.ckpt'))    

    XX = tf.reshape(states, [-1, state_dim]) 
    Y1l = tf.matmul(XX, aW1) 
    Y1 = tf.nn.relu(Y1l+aB1)
    Y2l = tf.matmul(Y1, aW2)
    Y2 = tf.nn.relu(Y2l+aB2)
    Ylogits = tf.matmul(Y2, aW3) + aB3
    out = tf.tanh(Ylogits)
    scaled_out = tf.multiply(out, action_bound)
    return scaled_out 

def predict(sess, states):
        return sess.run(actor(states, sess, True))


with tf.Session() as sess:
    
    _=actor([], sess, True)
    
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        
        for j in range(MAX_EP_STEPS): 
            #env.render()
            a = predict(sess, np.reshape(s, (1, state_dim)))
            print(a)
            s2, r, terminal, info = env.step(a[0])
        
            s = s2
            ep_reward += r
            
            if terminal:
                print ('| Reward: %.2i' % int(ep_reward), " | Episode", i) 
                break

if __name__ == '__main__':
  tf.app.run()
