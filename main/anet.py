# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:49:15 2017

@author: Kuanho

actor network
"""
####################################################################################################
#remember to make tst:True when all training done, and use the trained network to control
####################################################################################################
import tensorflow as tf
import numpy as np

class anet(object):
   def __init__(self, sess, s_dim, a_dim, action_bound, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_bound = action_bound
        self.tau = tau
       
        # variable learning rate
        self.lr = tf.placeholder(tf.float32)
       
        # train/test selector for batch normalisation
        self.tst = tf.placeholder(tf.bool)
        
        # training iteration
        self.iter = tf.placeholder(tf.int32)
        
        # Actor Network
        self.inputs, self.out, self.scaled_out, self.update_ema = self.create_a_net()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, self.target_update_ema = self.create_a_net()
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.lr).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
    
   #2 hidden layer_relu+BN_lrdecay_batchnorm
   def create_a_net(self):
       inputs = tf.placeholder(tf.float32, [None, self.s_dim])
       W1 = tf.Variable(tf.truncated_normal([self.s_dim, 400], stddev=0.001))
       B1 = tf.Variable(tf.ones([400])/1000)
       W2 = tf.Variable(tf.truncated_normal([400, 300], stddev=0.001))
       B2 = tf.Variable(tf.ones([300])/1000)
       W3 = tf.Variable(tf.truncated_normal([300, self.a_dim], stddev=0.001))
       B3 = tf.Variable(tf.zeros([self.a_dim]))
       XX = tf.reshape(inputs, [-1, self.s_dim]) 
       Y1l = tf.matmul(XX, W1)
       Y1bn, update_ema1 = self.batchnorm(Y1l, self.tst, self.iter, B1)
       Y1 = tf.nn.relu(Y1bn)
       Y2l = tf.matmul(Y1, W2)
       Y2bn, update_ema2 = self.batchnorm(Y2l, self.tst, self.iter, B2)
       Y2 = tf.nn.relu(Y2bn)
       Ylogits = tf.matmul(Y2, W3) + B3
       out = tf.tanh(Ylogits)
       scaled_out = tf.multiply(out, self.action_bound)
       update_ema = tf.group(update_ema1, update_ema2)
       return inputs, out, scaled_out, update_ema
  
   def batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
       exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
       bnepsilon = 1e-5
       if convolutional:
          mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
       else:
          mean, variance = tf.nn.moments(Ylogits, [0])
       update_moving_everages = exp_moving_avg.apply([mean, variance])
       m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
       v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
       Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
       return Ybn, update_moving_everages

   def no_batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
       return Ylogits, tf.no_op()
 
 
   def train(self, inputs, a_gradient, i):
     #learning rate decay
     max_learning_rate = 0.0001
     min_learning_rate = 0.000001
     decay_speed = 1000.0
     learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)  
     
     self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient, self.lr:learning_rate, self.tst: False
        })
     self.sess.run(self.update_ema, {self.inputs: inputs, 
                                   self.action_gradient: a_gradient, self.tst: False, self.iter: i})

   def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs, self.tst: False
        })

   def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs, self.tst: False
        })

   def update_target_network(self):
        self.sess.run(self.update_target_network_params)

   def get_num_trainable_vars(self):
        return self.num_trainable_vars
