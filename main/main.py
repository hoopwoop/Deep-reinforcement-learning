# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:03:35 2017

@author: Kuanho
Reference: Patrick Emami
main
"""
import tensorflow as tf
import gym
import os
import csv
import numpy as np
from gym import wrappers
from anet import anet
from cnet import cnet
from replay_buffer import ReplayBuffer
from OU import OUNoise

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 3000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
#ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
#CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001


# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = 'LunarLanderContinuous-v2'
# Directory for storing gym results
MONITOR_DIR = os.getcwd()+str('\\results\\gym_ddpg')
# Directory for storing tensorboard summary results
SUMMARY_DIR = os.getcwd()+str('\\results\\tf_ddpg')
# Pathy for storing model
MODEL_DIR = os.getcwd()+str('\\results\\model')
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Model save
# ===========================
def save_model(sess, actor_net, critic_net):
    anetf=open(MODEL_DIR+'\\actornet_weight', 'w')
    cnetf=open(MODEL_DIR+'\\criticnet_weight', 'w')
    #anetf.write("\n".join(map(lambda x: str(x), sess.run(actor_net))))
    #cnetf.write("\n".join(map(lambda x: str(x), sess.run(critic_net))))
    writera = csv.writer(anetf)
    writera.writerows(sess.run(actor_net))
    writerc = csv.writer(cnetf)
    writerc.writerows(sess.run(critic_net))
    anetf.close()
    cnetf.close()
    '''with open('actornet_weight', 'w') as outfile:
        json.dump(sess.run(actor_net), outfile)
    with open('criticnet_weight', 'w') as outfile:
        json.dump(sess.run(critic_net), outfile)'''
    print('''Model saved''')
    
    
# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic):
    ## Total steps
    TS=0
    ## Condition index
    CI=0
    
    #OU noise
    exploration_noise = OUNoise(actor.a_dim, mu=0, theta=0.15, sigma=0.07)
    
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EP_STEPS):

            if RENDER_ENV: 
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, env.observation_space.shape[0]))) + (1. / (1. + i + j)) 
            ##the addition makes action_value exceeds bound
            # random policy, if np.random.uniform(0,1)<2 means no random policy
            '''if np.random.uniform(0,1)<0.9:    
                a = actor.predict(np.reshape(s, (1, actor.s_dim)))
                

            else:
                a = np.array([np.random.uniform(-actor.action_bound, actor.action_bound) \
                                       for n in range(actor.a_dim)])  '''
            # Add OU noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + exploration_noise.noise()
    
            # Ensure the output is limited        
            a = np.minimum(np.maximum(a, -actor.action_bound), actor.action_bound)    
            
            s2, r, terminal, info = env.step(a[0])    
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a[0], (actor.a_dim,)), r, \
                terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:     
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k]*0.01)
                    else:
                        y_i.append(r_batch[k]*0.01 + GAMMA * target_q[k]) ## scale the reward

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)), TS)
            
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)                
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0], TS)

                # Update target networks
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

                break
            
        if CI >= 30:
            break
        
    save_model(sess, actor.target_net, critic.target_net)
    
def main(_):
    with tf.Session() as sess:
        
        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        #assert (env.action_space.high == -env.action_space.low)

        actor = anet(sess, state_dim, action_dim, action_bound, TAU)

        critic = cnet(sess, state_dim, action_dim, TAU, actor.get_num_trainable_vars(), MINIBATCH_SIZE)

        if GYM_MONITOR_EN:
            env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)
            '''if not RENDER_ENV:
                env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
            else:
                env.monitor.start(MONITOR_DIR, force=True)'''

        train(sess, env, actor, critic)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()