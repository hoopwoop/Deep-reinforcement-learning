# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:44:45 2017

@author: Kuanho

critic network
"""

####################################################################################################
#remember to make tst:True when all training done, and use the trained network to control
####################################################################################################
import tensorflow as tf
import numpy as np

class cnet(object):
    def __init__(self, sess, s_dim, a_dim, tau, num_actor_vars, MINIBATCH_SIZE):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.tau = tau
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        # variable learning rate
        self.lr = tf.placeholder(tf.float32)
       
        # train/test selector for batch normalisation
        self.tst = tf.placeholder(tf.bool)
        
        # training iteration
        self.iter = tf.placeholder(tf.int32)
        
        # Critic network
        self.inputs, self.action, self.out, self.update_ema = self.create_c_net()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out, self.target_update_ema = self.create_c_net()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch 
        # w.r.t. that action (i.e., sum of dy/dx over all ys). We then divide
        # through by the minibatch size to scale the gradients down correctly.
        ## cancel dividing batch size: self.action_grads = \
        ## tf.div(tf.gradients(self.out, self.action), tf.constant(MINIBATCH_SIZE, dtype=tf.float32))
        self.action_grads = tf.gradients(self.out, self.action)
        
    #2 hidden layer_relu+BN_lrdecay_batchnorm
    def create_c_net(self):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])
        #net = tflearn.fully_connected(inputs, 400, activation='relu')       
        W1 = tf.Variable(tf.truncated_normal([self.s_dim, 400], stddev=0.1))
        B1 = tf.Variable(tf.ones([400])/10)
       

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        
        #t1 = tflearn.fully_connected(net, 300)
        #t2 = tflearn.fully_connected(action, 300)
        W2s = tf.Variable(tf.truncated_normal([400, 300], stddev=0.1))
        W2a = tf.Variable(tf.truncated_normal([self.a_dim, 300], stddev=0.1))
        B2 = tf.Variable(tf.ones([300])/10)

        #net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        W3 = tf.Variable(tf.truncated_normal([300, 1], stddev=0.1))
        B3 = tf.Variable(tf.zeros([1]))
        XX = tf.reshape(inputs, [-1, self.s_dim]) 
        XXa = tf.reshape(action, [-1, self.a_dim])
        Y1l = tf.matmul(XX, W1)
        Y1bn, update_ema1 = self.batchnorm(Y1l, self.tst, self.iter, B1)
        Y1 = tf.nn.relu(Y1bn)
        Y2l = tf.matmul(Y1, W2s) + tf.matmul(XXa, W2a)
        Y2bn, update_ema2 = self.batchnorm(Y2l, self.tst, self.iter, B2)
        Y2 = tf.nn.relu(Y2bn)
        Ylogits = tf.matmul(Y2, W3) + B3
        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        #w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        #out = tflearn.fully_connected(net, 1, weights_init=w_init)
        out = Ylogits
        update_ema = tf.group(update_ema1, update_ema2)
        return inputs, action, out, update_ema

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
   
    def train(self, inputs, action, predicted_q_value, i):
      #learning rate decay
      max_learning_rate = 0.03
      min_learning_rate = 0.0001
      decay_speed = 1000.0
      learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)  
      return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value, self.lr:learning_rate, self.tst: False
        })
      self.sess.run(self.update_ema, {self.inputs: inputs, 
                                   self.action: action, self.tst: False, self.iter: i})


    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action, self.tst: False
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action, self.tst: False
        })

    def action_gradients(self, inputs, actions): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions , self.tst: False
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)