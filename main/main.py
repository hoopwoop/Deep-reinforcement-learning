# -*- coding: utf-8 -*-
"""
original author: Patrick Emami
author: kuanho
implementing ddpg
"""
import tensorflow as tf
import gym
import os
import csv
import numpy as np
from anet import anet
from cnet import cnet
from replay_buffer import ReplayBuffer
from OU import OUNoise

# training parameter
MAX_EPISODES = 2000
MAX_EP_STEPS = 1000
GAMMA = 0.99
TAU = 0.001

RENDER_ENV = False
GYM_MONITOR_EN = True
ENV_NAME = 'LunarLanderContinuous-v2'
MONITOR_DIR = os.path.join(os.getcwd(), 'results', 'gym_ddpg')
SUMMARY_DIR = os.path.join(os.getcwd(), 'results', 'tf_ddpg')
MODEL_DIR = os.path.join(os.getcwd(), 'results', 'model')
RANDOM_SEED = 1234
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 64

# save model
def save_model(sess, actor_net, critic_net):
    anetf=open(os.path.join(MODEL_DIR, 'actornet_weight_bias'), 'w')
    cnetf=open(os.path.join(MODEL_DIR, 'criticnet_weight_bias'), 'w')
    writera = csv.writer(anetf)
    writera.writerows(sess.run(actor_net))
    writerc = csv.writer(cnetf)
    writerc.writerows(sess.run(critic_net))
    anetf.close()
    cnetf.close()
    print('''Model saved''')
    
# save model_tensorflow
def save_model_tf(sess, actor_net, critic_net):   
    anet.update_target_network
    cnet.update_target_network
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(SUMMARY_DIR, 'model.ckpt'))
    print('''Model saved with tf''')

# summary    
def build_summaries(): 
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# train
def train(sess, env, actor, critic):
    # total steps
    TS=0
    
    # condition index
    CI=0
    
    # OU noise
    exploration_noise = OUNoise(actor.a_dim, mu=0, theta=0.15, sigma=0.05)
    
    # set summary ops
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # initialize target network
    actor.update_target_network()
    critic.update_target_network()

    # initialize replay buffer
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EP_STEPS):

            if RENDER_ENV: 
                env.render()          
                
            # add OU noise with inverse sigmoid decay
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + exploration_noise.noise()/(1+np.exp(0.1*i-30))
    
            # ensure the output is limited        
            a = np.minimum(np.maximum(a, -actor.action_bound), actor.action_bound)    
            
            # execute action
            s2, r, terminal, info = env.step(a[0])    
            
            # add experience
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a[0], (actor.a_dim,)), r, \
                terminal, np.reshape(s2, (actor.s_dim,)))

            # ensure buffer at least minibatch size to start training, and random sample
            if replay_buffer.size() > MINIBATCH_SIZE:     
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # target Q' with target action
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                # target value for critic net loss minimizing
                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k]*1)
                    else:
                        y_i.append(r_batch[k]*1 + GAMMA * target_q[k])

                # train critic
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)), TS)
            
                ep_ave_max_q += np.amax(predicted_q_value)

                # train actor
                a_outs = actor.predict(s_batch)                
                grads = critic.update_Q_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0], TS) 

                # update target networks
                actor.update_target_network()
                critic.update_target_network()
            

            s = s2
            ep_reward += r
            TS += 1
            

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print ('| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j)))
                
                if ep_reward >= 200: 
                    CI+=1
                    
                # reset noise    
                exploration_noise.reset()
                
                break
            
        
    #save model    
    save_model_tf(sess, actor.target_net, critic.target_net)
    
def main(_):
    with tf.Session() as sess:        
        # initial environment
        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        actor = anet(sess, state_dim, action_dim, action_bound, TAU)
        critic = cnet(sess, state_dim, action_dim, TAU, MINIBATCH_SIZE)

        if GYM_MONITOR_EN:
            env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        env.render.close(True)

if __name__ == '__main__':
    tf.app.run()