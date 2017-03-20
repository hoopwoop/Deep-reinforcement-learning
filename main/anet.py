# -*- coding: utf-8 -*-
"""
original author: Patrick Emami
author: kuanho
"""
import tensorflow as tf
import numpy as np

class anet(object):
   def __init__(self, sess, s_dim, a_dim, action_bound, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_bound = action_bound
        self.tau = tau
        self.lr = tf.placeholder(tf.float32)
       
        # iteration
        self.iter = tf.placeholder(tf.int32)
        
        # flag for exponential moving average
        self.tst = tf.placeholder(tf.bool)

        # actor network
        self.states, self.out, self.scaled_out, self.update_ema, self.net = self.create_a_net()

        # initialize target_net
        self.target_net = self.net
        
        # target network
        self.target_states, self.target_out, self.target_scaled_out, self.target_update_ema, self.target_net = self.create_a_target_net()

        # update target weight, and no need to update ema
        self.update_target = \
        [self.target_net[i].assign(tf.multiply(self.tau, self.net[i]) + tf.multiply((1-self.tau), self.target_net[i])) 
            for i in range(len(self.target_net)-2)]

        # initialize Q gradients
        self.Q_gradients = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # combine gradients, minus sugn because of tensorflow do descent but here needs ascend
        self.actor_gradients = tf.gradients(self.scaled_out, self.net[0:6], -self.Q_gradients)

        # optimize
        self.optimize = tf.train.AdamOptimizer(self.lr).\
            apply_gradients(zip(self.actor_gradients, self.net))
            
            
   # 2 hidden layer_relu with batchnorm
   def create_a_net(self):
       layer1_size = 400
       layer2_size = 300
       states = tf.placeholder(tf.float32, [None, self.s_dim])
       W1 = tf.Variable(tf.random_uniform([self.s_dim, layer1_size],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)))
       B1 = tf.Variable(tf.random_uniform([layer1_size],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)))
       W2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size],-1/np.sqrt(layer1_size),1/np.sqrt(layer1_size)))
       B2 = tf.Variable(tf.random_uniform([layer2_size],-1/np.sqrt(layer1_size),1/np.sqrt(layer1_size)))
       W3 = tf.Variable(tf.random_uniform([layer2_size, self.a_dim],-3e-3,3e-3))
       B3 = tf.Variable(tf.random_uniform([self.a_dim],-3e-3,3e-3))
       XX = tf.reshape(states, [-1, self.s_dim]) 
       Y1l = tf.matmul(XX, W1)
       Y1bn, update_ema1, ema1 = self.batchnorm(Y1l, self.tst, self.iter, B1) 
       Y1 = tf.nn.relu(Y1bn)
       Y2l = tf.matmul(Y1, W2)
       Y2bn, update_ema2, ema2 = self.batchnorm(Y2l, self.tst, self.iter, B2)
       Y2 = tf.nn.relu(Y2bn)
       Ylogits = tf.matmul(Y2, W3) + B3
       out = tf.tanh(Ylogits)
       scaled_out = tf.multiply(out, self.action_bound)
       update_ema = tf.group (update_ema1, update_ema2)
       return states, out, scaled_out, update_ema, [W1, B1, W2, B2, W3, B3, ema1, ema2]
   
    
    # target net
   def create_a_target_net(self):
       states = tf.placeholder(tf.float32, [None, self.s_dim])
       W1, B1, W2, B2, W3, B3, = self.target_net[0:6]
       XX = tf.reshape(states, [-1, self.s_dim]) 
       Y1l = tf.matmul(XX, W1)
       Y1bn, update_ema1, ema1 = self.batchnorm(Y1l, self.tst, self.iter, B1) 
       Y1 = tf.nn.relu(Y1bn)
       Y2l = tf.matmul(Y1, W2)
       Y2bn, update_ema2, ema2 = self.batchnorm(Y2l, self.tst, self.iter, B2)
       Y2 = tf.nn.relu(Y2bn)
       Ylogits = tf.matmul(Y2, W3) + B3
       out = tf.tanh(Ylogits)
       scaled_out = tf.multiply(out, self.action_bound)
       update_ema = tf.group (update_ema1, update_ema2)
       return states, out, scaled_out, update_ema, [W1, B1, W2, B2, W3, B3, ema1, ema2]
   
    
    # batchnorm
   def batchnorm(self, Ylogits, is_test, iteration, offset):
       bnepsilon = 1e-5
       exp_move_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
       mean, variance = tf.nn.moments(Ylogits, [0])
       ema = [exp_move_avg.average(mean), exp_move_avg.average(variance)]
       update_move_avg = exp_move_avg.apply([mean, variance])
       m = tf.cond(is_test, lambda: exp_move_avg.average(mean), lambda: mean)
       v = tf.cond(is_test, lambda: exp_move_avg.average(variance), lambda: variance)
       Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
       return Ybn, update_move_avg, ema
       
       
   def train(self, states, Q_gradients, i):
     learning_rate=0.0001
     self.sess.run(self.optimize, feed_dict={
            self.states: states,
            self.Q_gradients: Q_gradients, self.lr:learning_rate, self.tst: False, self.iter: i})
    
    # update moving average
     self.sess.run(self.update_ema, feed_dict={self.states: states, self.tst: False, self.iter: i})


   def predict(self, states, i):
        return self.sess.run(self.scaled_out, feed_dict={
            self.states: states, self.tst: True, self.iter: i
        })

   def predict_target(self, states, i):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_states: states, self.tst: True, self.iter: i
        })

   def update_target_network(self):
        self.sess.run(self.update_target)

