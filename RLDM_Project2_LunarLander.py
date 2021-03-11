#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import os
from collections import deque


# In[ ]:


from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow
tensorflow.compat.v1.disable_eager_execution()


# In[ ]:


env = gym.make('LunarLander-v2')
env.seed(42)
np.random.seed(42)
random.seed(42)
nS = env.observation_space.shape[0]
nA = env.action_space.n


# In[ ]:


n_episodes, n_steps, batch_size, rew, rew_band = 2000, 1000, 64, [], []
eps, eps_decay, eps_min   = 1.0, 0.995, 0.01
output_dir = 'model_output/lunarlander/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[ ]:


class DQNagent_LunarLander:
        def __init__(self, nS, nA):
            self.nS, self.nA  = nS, nA
            self.mem, self.gamma = deque(maxlen=100000), 0.99
            self.learn_rate = 0.0001
            self.Q_network = self.model_lander()
            self.target_network = self.model_lander()
            self.steps, self.freq = 0, 4
         
        def model_lander(self):
            model = Sequential()
            model.add(Dense(64, activation='relu',
                           input_dim=self.nS))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.nA, activation='linear'))
            model.compile(loss='mse',
                         optimizer=Adam(lr=self.learn_rate))
            return model

        def experience_add(self, state, action, reward, next_state, done):
            self.mem.append((state, action,
                               reward, next_state, done))
            
        def update_target_net(self):
            
            self.target_network.set_weights(self.Q_network.get_weights())  
            

        def Q_update(self, batch_size):
            
            self.steps += 1
            
            if self.steps % self.freq == 0:
                if len(self.mem) < batch_size: 
                    return
                
                batch = random.sample(self.mem, batch_size)
                
                states_bk, targets_bk = [], []
                for state, action, reward, next_state, done in batch:
                    z = reward
                    if not done:                    
                        z = reward +  (self.gamma * np.amax(self.target_network.predict(next_state)[0]))
                    target_val = self.Q_network.predict(state)
                    target_val[0][action] = z
                    states_bk.append(state[0])
                    targets_bk.append(target_val[0])
                             
                self.Q_network.fit(np.array(states_bk), np.array(targets_bk), epochs=1, verbose=0)
                self.update_target_net()
                

        def e_greedy(self, state, eps):
            if np.random.random() < eps:
                return np.random.randint(self.nA)
            return np.argmax(self.Q_network.predict(state)[0])

        def save_model(self, name):
            self.Q_network.save_weights(name)

        def load_model(self, name):
            self.Q_network.load_weights(name)


# In[ ]:


my_agent = DQNagent_LunarLander(nS, nA)


# In[ ]:


for e in range(n_episodes): 
    state = env.reset()
    state = np.reshape(state, [1, nS])
    done, rew_tot = False, 0

    for t in range(n_steps):
        action = my_agent.e_greedy(state, eps)
        next_state, reward, done, _ = env.step(action) 
        next_state = np.reshape(next_state, [1, nS])
        rew_tot += reward  
        my_agent.experience_add(state, action, reward, next_state, done)
        my_agent.Q_update(batch_size) 
        state = next_state   
        if done:  
            break
 
    rew.append(rew_tot)
    rew_band.append(rew_tot)
   
    if eps > eps_min:
        eps *= eps_decay
        
    if e % 100 == 0:
        print("\repisode {}\tAvg score: {:.2f}".format(e, np.mean(rew_band)))
        
    if np.mean(rew_band)>=200.0:
        print("\nsolved in {:d} episodes!\tAverage Score: {:.2f}".format(e, np.mean(rew_band)))
        my_agent.save_model(output_dir + "dqn_model.h5")
        break

