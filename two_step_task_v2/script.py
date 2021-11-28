import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gym
import agents   #MF and controller
import dqn_mb   # MB
#import ddqn
#import multistep
import random
import two_step_v4_5

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

random.seed(1)

x = np.arange((1000))   # number of episodes
total_iter =1000
remain_iter = total_iter


    
env = two_step_v4_5.Env()
env.rule_changer()
agent1 = agents.DQN(env)
agent2 = dqn_mb.DQN_MB(env)
controller = agents.Controller(thres_l=1.2, thres_h=2.5)      #adjust

while remain_iter > 0:

    # Begin with model-free
    # When controller says switch, learn model-free
    results, transfer_info, trained_iter = agent1.learn(remain_iter, controller)
    
    remain_iter-= trained_iter
    
    # When controller says switch, learn model-based
    results, trained_iter = agent2.learn(remain_iter, controller, transfer_info)
        
    remain_iter-= trained_iter
        
    #plt.plot(x, results, label='DQN')

del agent1, agent2

plt.plot(controller.uncertainty, label='DQN')

plt.plot(controller.flags, label='DQN')

print("Reinforcement Learning Finish")
print("Draw graph ... ")

plt.legend()
fig = plt.gcf()
plt.savefig("result.png")
plt.show()
