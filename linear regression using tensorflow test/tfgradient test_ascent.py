# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 23:22:44 2017

@author: Kuanho
"""

import numpy as np
import tensorflow as tf
# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(-(tf.square(W * x + b - y))) # sum of the squares
gradW=tf.gradients(loss,W,float(-2))[0]
gradb=tf.gradients(loss,b,float(-2))[0]
# optimizer
'''optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)'''
train=tf.train.AdamOptimizer(0.1).apply_gradients([(gradW,W),(gradb,b)])
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
g_W, g_b =sess.run([gradW,gradb], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
print("gradW: %s gradb: %s"%(g_W, g_b))