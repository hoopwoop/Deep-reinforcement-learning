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
        
         # iteration
        self.iter = tf.placeholder(tf.int32)
        
        # flag for exponential moving average
        self.tst = tf.placeholder(tf.bool)
        
        # critic network
        self.states, self.actions, self.out, self.update_ema, self.net = self.create_c_net()
        
        # initialize target_net
        self.target_net = self.net
        
        # target network
        self.target_states, self.target_actions, self.target_out, self.target_update_ema, self.target_net = self.create_c_target_net()

        # update target weight, and no need to update ema
        self.update_target = \
        [self.target_net[i].assign(tf.multiply(self.tau, self.net[i]) + tf.multiply((1-self.tau), self.target_net[i])) 
            for i in range(0,13)]
    
        # initialize target value for critic net loss minimizing
        self.y = tf.placeholder(tf.float32, [None, 1])

        # loss $ optimization
        self.loss = tf.reduce_mean(tf.square(self.y - self.out))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        # calculate Q gradients
        # do average over batch here
        self.Q_gradients = tf.divide(tf.gradients(self.out, self.actions), tf.constant(self.MINIBATCH_SIZE, tf.float32))
        
        
    # 2 hidden layer_relu with batchnorm, and action input add in at 2nd layer
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
        O0 = tf.Variable(tf.random_uniform([self.s_dim],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)))
        S0 = tf.Variable(tf.random_uniform([self.s_dim],-1/np.sqrt(self.s_dim),1/np.sqrt(self.s_dim)))
        O0a = tf.Variable(tf.random_uniform([self.a_dim],-1/np.sqrt(self.a_dim),1/np.sqrt(self.a_dim)))
        S0a = tf.Variable(tf.random_uniform([self.a_dim],-1/np.sqrt(self.a_dim),1/np.sqrt(self.a_dim)))
        O3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))
        S3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))
        XX = tf.reshape(states, [-1, self.s_dim]) 
        XXbn, update_ema0, ema0 = self.batchnorm(XX, self.tst, self.iter, O0, S0)
        XXa = tf.reshape(actions, [-1, self.a_dim])
        XXabn, update_ema0a, ema0a = self.batchnorm(XXa, self.tst, self.iter, O0a, S0a)
        Y1l = tf.matmul(XXbn, W1)
        Y1bn, update_ema1, ema1 = self.batchnorm(Y1l, self.tst, self.iter, B1, None)
        Y1 = tf.nn.relu(Y1bn)
        Y2l = tf.matmul(Y1, W2s) + tf.matmul(XXabn, W2a)
        Y2bn, update_ema2, ema2 = self.batchnorm(Y2l, self.tst, self.iter, B2, None)
        Y2 = tf.nn.relu(Y2bn)
        Y3l = tf.matmul(Y2, W3)
        Y3bn, update_ema3, ema3 = self.batchnorm(Y3l, self.tst, self.iter, O3, S3)
        out = tf.identity(Y3bn + B3)
        update_ema = tf.group (update_ema0, update_ema0a, update_ema1, update_ema2, update_ema3)
        return states, actions, out, update_ema, [W1, B1, W2s, W2a, B2, W3, B3, O0, S0, O0a, S0a, O3, S3, ema0, ema0a, ema1, ema2, ema3]
    
    
    # target net
    def create_c_target_net(self):
        states = tf.placeholder(tf.float32, [None, self.s_dim])
        actions = tf.placeholder(tf.float32, [None, self.a_dim])      
        W1, B1, W2s, W2a, B2, W3, B3, O0, S0, O0a, S0a, O3, S3 = self.target_net[0:13]
        XX = tf.reshape(states, [-1, self.s_dim]) 
        XXbn, update_ema0, ema0 = self.batchnorm(XX, self.tst, self.iter, O0, S0)
        XXa = tf.reshape(actions, [-1, self.a_dim])
        XXabn, update_ema0a, ema0a = self.batchnorm(XXa, self.tst, self.iter, O0a, S0a)
        Y1l = tf.matmul(XXbn, W1)
        Y1bn, update_ema1, ema1 = self.batchnorm(Y1l, self.tst, self.iter, B1, None)
        Y1 = tf.nn.relu(Y1bn)
        Y2l = tf.matmul(Y1, W2s) + tf.matmul(XXabn, W2a)
        Y2bn, update_ema2, ema2 = self.batchnorm(Y2l, self.tst, self.iter, B2, None)
        Y2 = tf.nn.relu(Y2bn)
        Y3l = tf.matmul(Y2, W3)
        Y3bn, update_ema3, ema3 = self.batchnorm(Y3l, self.tst, self.iter, O3, S3)
        out = tf.identity(Y3bn + B3)
        update_ema = tf.group (update_ema0, update_ema0a, update_ema1, update_ema2, update_ema3)
        return states, actions, out, update_ema, [W1, B1, W2s, W2a, B2, W3, B3, O0, S0, O0a, S0a, O3, S3, ema0, ema0a, ema1, ema2, ema3]

   
    # batchnorm
    def batchnorm(self, Ylogits, is_test, iteration, offset, scale):
        bnepsilon = 1e-5
        exp_move_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        mean, variance = tf.nn.moments(Ylogits, [0])
        ema = [exp_move_avg.average(mean), exp_move_avg.average(variance)]
        update_move_avg = exp_move_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_move_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_move_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, scale, bnepsilon)
        return Ybn, update_move_avg, ema
   
    
    def train(self, states, actions, y, i):
      learning_rate=0.001
      return self.sess.run([self.out, self.optimize], feed_dict={
            self.states: states,
            self.actions: actions,
            self.y: y, self.lr:learning_rate, self.tst: False, self.iter: i
        })
    
      # update moving average
      self.sess.run(self.update_ema, feed_dict={self.states: states, self.actions: actions, self.tst: False, self.iter: i})
      self.sess.run(self.target_update_ema, feed_dict={self.target_states: states, self.target_actions: actions, self.tst: False, self.iter: i})


    def predict(self, states, actions, i):
        return self.sess.run(self.out, feed_dict={
            self.states: states,
            self.actions: actions, self.tst: True, self.iter: i
        })

    def predict_target(self, states, actions, i):
        return self.sess.run(self.target_out, feed_dict={
            self.target_states: states,
            self.target_actions: actions, self.tst: True, self.iter: i
        })

    def update_Q_gradients(self, states, actions, i): 
        return self.sess.run(self.Q_gradients, feed_dict={
            self.states: states,
            self.actions: actions, self.tst: True, self.iter: i
        })

    def update_target_network(self):
        self.sess.run(self.update_target)