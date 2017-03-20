# -*- coding: utf-8 -*-
"""
original author: Patrick Emami
author: kuanho
"""
import tensorflow as tf
import numpy as np

class cnet(object):
    def __init__(self, sess, s_dim, a_dim, tau, MINIBATCH_SIZE):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.tau = tau
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.lr = tf.placeholder(tf.float32)
        
        # critic network
        self.states, self.actions, self.out, self.net = self.create_c_net()
        
        # initialize target_net
        self.target_net = self.net
        
        # target network
        self.target_states, self.target_actions, self.target_out, self.target_net = self.create_c_target_net()

        # update target weight
        self.update_target = \
        [self.target_net[i].assign(tf.multiply(self.tau, self.net[i]) + tf.multiply((1-self.tau), self.target_net[i])) 
            for i in range(len(self.target_net))]
    
        # initialize target value for critic net loss minimizing
        self.y = tf.placeholder(tf.float32, [None, 1])

        # loss $ optimization
        self.loss = tf.reduce_mean(tf.square(self.y - self.out))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        # calculate Q gradients
        # do average over batch here
        self.Q_gradients = tf.divide(tf.gradients(self.out, self.actions), tf.constant(self.MINIBATCH_SIZE, tf.float32))
        
        
    # 2 hidden layer_relu, and action input add in at 2nd layer
    def create_c_net(self):
        layer1_size = 400
        layer2_size = 300        
        states = tf.placeholder(tf.float32, [None, self.s_dim])
        actions = tf.placeholder(tf.float32, [None, self.a_dim])     
        W1 = tf.Variable(tf.random_uniform([self.s_dim, layer1_size],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)))
        B1 = tf.Variable(tf.random_uniform([layer1_size],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)))
        W2s = tf.Variable(tf.random_uniform([layer1_size, layer2_size],-1/np.sqrt(layer1_size + self.a_dim),1/np.sqrt(layer1_size + self.a_dim)))
        W2a = tf.Variable(tf.random_uniform([self.a_dim, layer2_size],-1/np.sqrt(layer1_size + self.a_dim),1/np.sqrt(layer1_size + self.a_dim))) 
        B2 = tf.Variable(tf.random_uniform([layer2_size],-1/np.sqrt(layer1_size + self.a_dim),1/np.sqrt(layer1_size + self.a_dim)))
        W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
        B3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))
        XX = tf.reshape(states, [-1, self.s_dim]) 
        XXa = tf.reshape(actions, [-1, self.a_dim])
        Y1l = tf.matmul(XX, W1)
        Y1 = tf.nn.relu(Y1l+B1)
        Y2l = tf.matmul(Y1, W2s) + tf.matmul(XXa, W2a)
        Y2 = tf.nn.relu(Y2l+B2)
        Ylogits = tf.identity(tf.matmul(Y2, W3) + B3)
        out = Ylogits
        return states, actions, out, [W1, B1, W2s, W2a, B2, W3, B3]
    
    
    # target net
    def create_c_target_net(self):
        states = tf.placeholder(tf.float32, [None, self.s_dim])
        actions = tf.placeholder(tf.float32, [None, self.a_dim])      
        W1, B1, W2s, W2a, B2, W3, B3 = self.target_net
        XX = tf.reshape(states, [-1, self.s_dim]) 
        XXa = tf.reshape(actions, [-1, self.a_dim])
        Y1l = tf.matmul(XX, W1)
        Y1 = tf.nn.relu(Y1l+B1)
        Y2l = tf.matmul(Y1, W2s) + tf.matmul(XXa, W2a)
        Y2 = tf.nn.relu(Y2l+B2)
        Ylogits = tf.identity(tf.matmul(Y2, W3) + B3)
        out = Ylogits
        return states, actions, out, [W1, B1, W2s, W2a, B2, W3, B3]

   
    def train(self, states, actions, y, i):
      learning_rate=0.001
      return self.sess.run([self.out, self.optimize], feed_dict={
            self.states: states,
            self.actions: actions,
            self.y: y, self.lr:learning_rate
        })


    def predict(self, states, actions):
        return self.sess.run(self.out, feed_dict={
            self.states: states,
            self.actions: actions
        })

    
    def predict_target(self, states, actions):
        return self.sess.run(self.target_out, feed_dict={
            self.target_states: states,
            self.target_actions: actions
        })

    
    def update_Q_gradients(self, states, actions): 
        return self.sess.run(self.Q_gradients, feed_dict={
            self.states: states,
            self.actions: actions
        })

    
    def update_target_network(self):
        self.sess.run(self.update_target)