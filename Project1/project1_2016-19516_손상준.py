#!/usr/bin/env python
# coding: utf-8

# # Project 1. HMM 적용하여 데이터 모델링 해보기
# 
# ## Hidden Markov Model
# 
# * 은닉마코프모델 계산 및 구현
#   * https://ratsgo.github.io/machine%20learning/2017/10/14/computeHMMs/
# * https://web.stanford.edu/~jurafsky/slp3/A.pdf
#   
# ## Character Trajactories Data Set
# * Import mat-formatted data
#   * `mixoutALL_shifted.mat`
# * Data Features
#   * Data Set Characteristics: Time-Series
#   * Number of Instances: 2858
#   * Attribute Characteristics: Real
#   * Number of Attributes: 3
#   * Missing Values? N/A
#   * Each character sample is a 3-dimensional pen tip velocity trajectory. This is contained in matrix format, with 3 rows and T columns where T is the length of the character sample.
#   * 3 Dimensions were kept - x, y, and pen tip force.
#   * http://archive.ics.uci.edu/ml/datasets/Character+Trajectories

# In[1]:


from scipy import io
import matplotlib.pyplot as plt
import numpy as np

mat_file = io.loadmat('./mixoutALL_shifted.mat')

# print(mat_file['consts'].dtype)
consts = mat_file['consts'][0,0]
mixouts = mat_file['mixout'][0]

keys = np.squeeze(consts['key'][0])
charlabels = consts['charlabels'][0] - 1

num_keys = keys.size
num_mixouts = mixouts.shape[0]
T = 0

print('num_mixouts:', num_mixouts)
print('num_keys:', num_keys)
for i in range(num_keys):
    print(keys[i], end='')

for i in range(num_mixouts):
    mixouts[i] = mixouts[i].T
    mask = (mixouts[i][:,0] != 0) | (mixouts[i][:,1] != 0)
    mixouts[i] = mixouts[i][mask]
    T = max(T, mixouts[i].shape[0])

# print()
# for i in range(len(charlabels)):
#     print(i, charlabels[i])


# In[2]:


def plot_disp(disp, time):
    plt.scatter(disp[:time//3,0], disp[:time//3,1], color='blue')
    plt.scatter(disp[time//3:time*2//3,0], disp[time//3:time*2//3,1], color='green')
    plt.scatter(disp[time*2//3:,0], disp[time*2//3:,1], color='red')
#     plt.show()
    
def plot_vel_acc(vel, acc, time):
    plt.subplot('121')
    plt.scatter(vel[:time//3,0], vel[:time//3,1], color='blue')
    plt.scatter(vel[time//3:time*2//3,0], vel[time//3:time*2//3,1], color='green')
    plt.scatter(vel[time*2//3:,0], vel[time*2//3:,1], color='red')
    plt.subplot('122')
    plt.scatter(acc[:time//3,0], acc[:time//3,1], color='blue')
    plt.scatter(acc[time//3:time*2//3,0], acc[time//3:time*2//3,1], color='green')
    plt.scatter(acc[time*2//3:,0], acc[time*2//3:,1], color='red')
    plt.show()    
    
def plot_acc_angle(acc_angle, time):
    plt.plot(range(0,time//3),acc_angle[:time//3], color='blue')
    plt.plot(range(time//3,time*2//3),acc_angle[time//3:time*2//3], color='green')
    plt.plot(range(time*2//3,time),acc_angle[time*2//3:], color='red')
    plt.show()
    
def plot_obs(obs, time):
    plt.plot(range(0,time//3),obs[:time//3], color='blue')
    plt.plot(range(time//3,time*2//3),obs[time//3:time*2//3], color='green')
    plt.plot(range(time*2//3,time),obs[time*2//3:], color='red')
    plt.show()


# In[17]:


def get_observation(disp):
    time = disp.shape[0]
    vel = np.zeros((time, 2))
    acc = np.zeros((time, 2))
    
    vel_angle = np.zeros(time)
    angle = np.zeros(time)
    obs = np.zeros(time, dtype=int)
    for i in range(1, time):
        vel[i] = disp[i] - disp[i-1]
        acc[i] = vel[i] - vel[i-1]
        vel_angle[i] = np.arctan2(vel[i][0], vel[i][1])
        if vel_angle[i] < 0:
            vel_angle[i] += 2 * np.pi
        angle[i] = np.arctan2(acc[i][0], acc[i][1])
        if angle[i] < 0:
            angle[i] += 2 * np.pi
        obs[i] = int((angle[i] / np.pi * 3) % 6) + int((vel_angle[i] / np.pi * 2) % 4) * 6
    return (time, vel, acc, angle, obs)

num_features = 24
features = np.empty(num_mixouts, dtype=object)
prev = None
for i in range(0, num_mixouts, 1):
    disp = mixouts[i][:,:2]
    (time, vel, acc, angle, obs) = get_observation(disp)
    features[i] = obs
    
    if prev != charlabels[i]:
        print(i, keys[charlabels[i]])
        prev = charlabels[i]
        disp = mixouts[i][:,:2]
        
#         plt.subplot('121')
#         plot_disp(disp, time)
        
#         plt.subplot('122')
#         plot_obs(obs, time)
        
#         plot_vel_acc(vel, acc, time)
#         plot_acc_angle(angle, time)


# ## train_test_split
# * `train_size : test_size = 90 : 10`

# In[13]:


from sklearn.model_selection import train_test_split

# train, test split
data = np.array([[charlabels[i], features[i][:], i] for i in range(num_mixouts)])
train_data, test_data = train_test_split(data, test_size=0.1)

# sort in charlabel
train_data = train_data[train_data[:, 0].argsort()]
test_data = test_data[test_data[:, 0].argsort()]

# print('train:', train_data.shape)
# print('test:', test_data.shape)
# print(test_data)


# # Character Recognition HMM
# ## HMM Learn
# * `hmmlearn` Tutorial
#   * https://hmmlearn.readthedocs.io/en/latest/tutorial.html
# * MultinomialHMM API Reference
#   * https://hmmlearn.readthedocs.io/en/latest/api.html#multinomialhmm

# In[5]:


from hmmlearn import hmm
import time

models = np.empty([num_keys], dtype=object)
idx = np.zeros(num_keys + 1, dtype=int)
for i in range(num_keys):
    models[i] = hmm.MultinomialHMM(n_components=num_features, verbose=False, n_iter=1)

for i in range(train_data.shape[0]):
    idx[train_data[i][0]+1] = i+1;

# multinomial HMM learn
for key in range(num_keys):
    start_time = time.time()
    trainRange = range(idx[key], idx[key+1])
    print('Training', keys[key], 'model w.', idx[key+1]-idx[key], 'examples', end=' ')

    trainX = np.concatenate([train_data[i][1].reshape(-1, 1) for i in trainRange])
    lengths = [len(train_data[i][1]) for i in trainRange]

    models[key].fit(trainX, lengths)
    print("(elapsed time: {}s).".format(time.time() - start_time))
        


# In[6]:


testSize = test_data.shape[0]
wrongCases = 0

print('Wrong Cases')
for i in range(testSize):
    testX = np.concatenate([test_data[i][1].reshape(-1, 1)])
    
    maxScore = models[0].score(testX)
    maxKey = 0
    
    for key in range(num_keys):
        score = models[key].score(testX)
        if maxScore < score:
            maxScore = score
            maxKey = key
            
#     print(keys[test_data[i][0]], keys[maxKey], maxScore, test_data[i][0] == maxKey)
    if not test_data[i][0] == maxKey:
        wrongCases += 1
        print(keys[test_data[i][0]], keys[maxKey], maxScore)
        disp1 = mixouts[test_data[i][2]][:,:2]
        disp2 = mixouts[train_data[idx[maxKey]][2]][:,:2]
        
#         plt.subplot('121')
#         plot_disp(disp1, disp1.shape[0])
        
#         plt.subplot('122')
#         plot_disp(disp2, disp2.shape[0])
#         plt.show()
print('(Wrong, Total)', (wrongCases, testSize))
print('Accuracy: {}%'.format((1 - wrongCases/testSize) * 100))


# In[ ]:




