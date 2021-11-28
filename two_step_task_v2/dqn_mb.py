import sys
import numpy as np
import copy
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import random
import gym
from acme.utils import tree_utils
from collections import deque
import collections

# BATCH_SIZE = 5
# LEARNING_RATE = 0.004         # learning rate parameter
# LEARNING_STARTS = 2000        # 1000 스텝 이후 training 시작
# GAMMA = 0.98                  # discount reward
# TARGET_UPDATE = 600           # target network update interval
# TOTAL_STEP_COUNT = 0 

FLAG = True

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque()
        #self.batch_size = 32
        self.size_limit = 30000
        self.Transitions = collections.namedtuple('Transitions',
                                                  ['state', 'action', 'reward', 'next_state', 'done_mask'])   #'discount'
        
    
    def put(self, state, action, reward, next_state, done_mask):  #discount
        transition = self.Transitions(
            state=state,
            action=action,
            reward=reward,
            #discount=discount,
            next_state=next_state,
            done_mask=done_mask
            )
        self.buffer.append(transition)
        
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # Convert the list of `batch_size` Transitions into 
        # a single Transitions object where each field has `batch_size` stacked fields.
        return tree_utils.stack_sequence_fields(batch)
    
    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self):
        
        super(Qnet,self).__init__()
        self.fc1 = nn.Linear(1,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,4)
        
        
    def forward(self,x):
        
        #print('x',x)
        x = torch.unsqueeze(x, -1).float()
        #print('x',x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print('x:', x)
        return x
        
    def sample_action(self, obs, epsilon):
        
        obs = torch.from_numpy(np.array(int(obs)))
        #print('obs',obs)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            #print('random action!')
            return random.randint(0,3)
        else :
            return out.argmax().item()
        
class DQN_MB:
    
    def __init__(self, env, batch_size=5, learning_rate=0.004, 
                 target_update_frequency=600, gamma=0.98):

        # Set environment
        self.env = env
        # self.env.rule_changer()
        self.env.reset()
        
        self.state_size = 2
        #print('state size:', self.state_size)
        self.action_size = 2
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.q= Qnet()
        #second qnet with same structure and initial values, updated separately
        self.q_target = copy.deepcopy(self.q)
        # self.q_target = Qnet()
        # self.q_target.load_state_dict(self.q.state_dict())
        
        self.target_update_frequency = target_update_frequency
        self.total_step_count = 0
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.q.parameters(), lr = learning_rate)
        self.loss_fn = F.smooth_l1_loss

        
    # def predict(self, state):
    #     # state를 넣어 policy에 따라 action을 반환
    #     return self.env.action_space.sample()

    def optimize_model(self):

        for i in range(10):
            
            # Sample a minibatch of transitions from experience replay (list)
            transitions = self.memory.sample(self.batch_size)
           # print("sample batch")

            # shape [batch_size, ...]
            s = torch.tensor(transitions.state)
            a = torch.tensor(transitions.action, dtype=torch.int64)
            r = torch.tensor(transitions.reward)
            #d = torch.tensor(transitions.discount)
            s_prime = torch.tensor(transitions.next_state)
            done_mask = torch.tensor(transitions.done_mask)

            # (1) Compute Q-values at original state
            # s shape [batch_size,state_size]
            # q(s) shape [batch_size,action_size]
            q_s = self.q(s) 
            
            # (2) Gather the Q-value corresponding to each action in the batch.
            q_s_a = q_s.gather(1, a.view(-1,1)).squeeze(0)   #q_s.gather(1,a)
            
            
            # (3) Compute Q-values at next states in the batch 
            # (unsqueeze to match dimensions)
            max_q_s_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            
            #print('max_q_prime:', max_q_prime)
            
            # (4) Compute TD error
            target_q = r + self.gamma * max_q_s_prime   #* done_mask

            
            loss = self.loss_fn(q_s_a, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            #print('i:',i)
            #print('backward loss update!')
            self.optimizer.step()
            

    def learn(self, max_episode, controller, transfer_memory):
        
        self.memory = transfer_memory
        #print('memory',self.memory.size())

        episode_record = []     # plot을 그리기 위해 데이터를 저장하는 list
        last_10_game_reward = deque(maxlen=30)
        
        #win_num = 0 
        reward_sum = 0
        print("=" * 70)
        print("begin mb learn")
        
        try:
            for episode in range(max_episode):
                #print("episode:",episode)
                
                done = False
                state = self.env.reset()
                step_count = 0
                
                #epsilon = max(0.01, 0.1 - 0.1*1*(episode/max_episode))
                epsilon = 1
    
                # episode 시작
                while not done:
                    if episode > 100 : self.env.render(state, reward_sum, episode)
                    
                    # Select epsilon-greedy action
                    action = self.q.sample_action(state, epsilon)
                    #action = self.predict(state)
                    
                    if action == 0:
                        action1 = 0
                        action2 = 0
                    
                    elif action == 1:
                        action1 = 0
                        action2 = 1
                    
                    elif action == 2:
                        action1 = 1
                        action2 = 0
                    
                    elif action == 3:
                        action1 = 1
                        action2 = 1
                    
                    #if episode < 100:
                    #    action = random.randint(0,2)
                    #print('state',state)
                    #이렇게 나온 action을 적용한 결과가 next_state 
                    next_state1, reward, done = self.env.step(state, action1)
                    state = next_state1
                    next_state, reward, done = self.env.step(state, action2)
                    reward_sum += reward
    
                    state = 0

                    done_mask = 0.0 if done else 1.0
                    
                    print('state:{}, action:{}, reward:{}, next_state:{}'.format(state, action1, reward, next_state1))
                    self.memory.put(state, action, reward, next_state, done_mask)
                    
                    # Controller checks for model switch in the beginning of every trial
                    if state == 0 :
                        print("call controller")
                        
                        #self.controller.put(action, next_state)
                        controller.put(action1, next_state1)
                        switch = controller.switch_model()
                        print('switch', switch)
                        if switch == 1:
                            print("SWITCH to model-free!")
                            # save info to transfer
                            #self.transfer_info = self.memory   # class ReplayBuffer
                            raise StopIteration        
                    #print('memory_size:' , self.memory.size())
                    
                    
                    state = next_state
                    step_count += 1
                    self.total_step_count += 1
                    
                    #print('TOTAL_STEP_COUNT:',TOTAL_STEP_COUNT)
    
                # Record the average reward of recent 100 episodes
                #print('last_10_game_reward',last_10_game_reward)
                last_10_game_reward.append(reward_sum)
                reward_sum = 0 
                avg_reward = np.mean(last_10_game_reward)
                episode_record.append(avg_reward)
                print("[Episode {:>5}]  episode steps: {:>5} avg: {}".format(episode, step_count, avg_reward))
                
                # no need to wait b/c transfer memory
                if self.memory.size() > 30:
                        self.optimize_model()
                
                if self.total_step_count % self.target_update_frequency == 0 : 
                    print('++++++++++copy weight from target network+++++++++++++')
                    self.q_target.load_state_dict(self.q.state_dict())
        except StopIteration : pass    
            
        #print('win_num:', win_num)              
        return episode_record, episode

