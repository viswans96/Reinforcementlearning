import numpy as np
import matplotlib.pyplot as plt
import random


# In[22]:


class RandomWalk:
    def __init__(self):
        self.training_sets = 100
        self.train_seq  = 10
        self.ideal_weights = [1/6, 1/3, 1/2, 2/3, 5/6]
        self.alpha = 0.01
        self.ep1 = 0.001
        
        
    def generate(self):
        seq, end = [3], [0, 6] 
        while seq[-1] not in end:
            step, randn = seq[-1], random.choice([1, -1])
            seq.append(step + randn)
        return seq
    
    def generate_trainsets(self):
        train_sets  = [] 
        random.seed(10)
        train_sets = [[self.generate() for i in range(self.train_seq)] for i in range(self.training_sets)]
        return train_sets   
    
    def td_updt(self, lambdas, alpha, val, seq):
        updt, elig = np.zeros(7), np.zeros(7)
        length = len(seq) - 1
        for i in range(0, length):
            nxt_st, cur_st = seq[i + 1], seq[i]
            td = alpha * (val[nxt_st] - val[cur_st])
            elig[cur_st] += 1.0
            updt = (td * elig) + updt
            elig = lambdas * elig
        return updt
    
    def train(self, lambdas, train_set):
        for training_set in train_set:
            values = np.zeros(7)
            while True:
                weight_old, delta_weight  = np.copy(values), np.zeros(7)
                values[6] = 1.0
                for sequence in training_set:
                    delta_weight += self.td_updt(lambdas, self.alpha, values, sequence) # compute delta_weight over all sequences in a set  
                values += delta_weight # update values after a training set is processed  
                if np.sum(np.abs(weight_old - values)) <= self.ep1: 
                    break
        predicted = np.array(values[1:-1])
        return np.sqrt(np.mean(np.power((predicted - self.ideal_weights), 2)))
    
    def train_eachseq(self, alpha, lambdas, train_set):
        rmse = 0 
        for training_set in train_set:
            values = np.array([0.5 for i in range(7)])
            for sequence in training_set:
                values[0], values[6] = 0.0, 1.0
                values += self.td_updt(lambdas, alpha, values, sequence)
                predicted = np.array(values[1:-1])
                rmse += np.sqrt(np.mean(np.power((predicted - self.ideal_weights), 2)))
        return rmse/ (self.train_seq * self.training_sets) 


# In[23]:


exp = RandomWalk()


# In[25]:


train_set = exp.generate_trainsets() 


# In[26]:


#Figure 3
lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
rmses = []
for l in lambdas:
    rmse = exp.train(l, train_set)
    rmses.append(rmse)

plt.figure(figsize=(6, 4))
plt.xlabel('λ',size=18)
plt.ylabel('ERROR', size = 15)
plt.title('Figure 3', size = 15)
plt.annotate('Widrow-Hoff',(0.7,rmses[-1]),fontsize=15)
plt.plot(lambdas,rmses,'-o')
plt.show()


# In[27]:


#Figure 4
alphas = np.linspace(0, 0.7, 15)
lambdas = [0.0, 0.3, 0.8, 1.0]
rmses_inc = {0.0: [], 0.3: [], 0.8: [], 1.0: []}
    
for l in lambdas:
    for a in alphas:
        rmse = exp.train_eachseq(a, l, train_set)
        rmses_inc[l].append(rmse)    

plt.figure(figsize=(6, 4))
for l in lambdas:
    plt.plot(alphas, rmses_inc[l], '-o', label='λ = {}'.format(l))
    plt.annotate('λ = {}'.format(l), (0.64, rmses_inc[l][-1]))
plt.xlabel('α',size=18)
plt.ylabel('ERROR', size = 15)
plt.ylim(0, 1.2)
plt.yticks(np.linspace(0, 1.2, 7))
plt.xlim(-0.10, 0.8)
plt.xticks(np.linspace(0, 0.6, 7))
plt.title('Figure 4', size = 15)
plt.legend()
plt.show()


# In[28]:


#Figure 5
best_alpha = 0.3
lambdas = np.linspace(0,0.7,15)
rmses = []
for l in lambdas:
    rmse = exp.train_eachseq(best_alpha, l, train_set)
    rmses.append(rmse)

plt.figure(figsize=(6,4))
plt.plot(lambdas,rmses,'-o')
plt.title('Figure 5',size=15)
plt.xlabel('λ',size=18)
plt.ylabel('ERROR USING BEST α',size=15)
plt.annotate('Widrow-Hoff',(0.70,rmses[-1]),fontsize=12)
plt.show()