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

#BATCH_SIZE = 32
#LEARNING_RATE = 0.004         # learning rate parameter 0.4
#LEARNING_STARTS = 2000        # start training after n trials
#GAMMA = 0.98                  # discount reward
#TARGET_UPDATE_FREQ = 600      # target network update interval
#TOTAL_STEP_COUNT = 0 
                 # mode 0 = mf, 1 = mb
 
#%% Controller of shifting model
class Controller():
    def __init__(self, n_states=2, n_actions=2, buffer_capacity=10, thres_l=0.01, thres_h=0.03):
        # state : stage1
        # action1, action2 : 0 = left, 1 = right
        self.n_states = n_states
        self.n_actions = n_actions
        
        # buffer to save next_state based on state and action 
        self.buffer_capacity= buffer_capacity
        self.trans_buffer = []
        self.trans_probs = []
        self.update = []
        self.entropy = 0
        
        self.FLAG = 0
        self.flags = []
        self.uncertainty = []
        
        #shape [4 x buffer_capacity]
        for _ in range(n_states*n_actions):
            self.trans_buffer.append(deque(maxlen=buffer_capacity))
            self.trans_probs.append(deque(maxlen=buffer_capacity))
            self.update.append(deque(maxlen=buffer_capacity))
            
        # initialize probs with 0.5
        for i in range(n_states*n_actions) : self.trans_probs[i].append(0.5) 
        
        #self.update = np.empty((n_states*n_actions,))
        self.k = 0
        self.thres_l = thres_l
        self.thres_h = thres_h
        
    def put(self, action, next_state):   # to include reward probabilities, add state
        # save memory
        
        for i in range(self.n_states*self.n_actions):
            if len(self.trans_buffer[i]) > self.buffer_capacity:
                self.trans_buffer[i].popleft()      
        
        #print('nextstate', next_state)       
        if action == 0 and next_state == 1: i = 0
        elif action == 1 and next_state == 1: i = 1
        elif action == 0 and next_state == 2: i = 2
        elif action == 1 and next_state == 2: i = 3
 
        for j in range(self.n_states*self.n_actions):    
            if j == i : self.trans_buffer[j].append(1)
            elif j%2 == i%2 : self.trans_buffer[j].append(0)
            
        
    def calculate_prob_update(self, lr=0.2):

        for i in range(self.n_states*self.n_actions):
            
            print(self.trans_buffer[i])
            prob = np.mean(self.trans_buffer[i])  
            print('prob:', prob)
            
            #entropy
            if prob == 0.0 : self.entropy = 6
            else : self.entropy= np.log2(1/prob)
            
            self.update[i] = self.entropy 
            
            # self.trans_probs[i].append(prob)
            
            # self.update[i] = np.abs(np.diff(self.trans_probs[i]))
            
            if len(self.trans_probs[i]) > self.buffer_capacity+1:
                self.trans_probs[i].popleft()            
        
        # mean of update of probabilities across types and trials
        k = np.mean(self.update)
        
        return k
        
    def switch_model(self):
        k = self.calculate_prob_update()
        print("k:", k)
        
        self.uncertainty.append(k)
        self.flags.append(self.FLAG)
        
        if self.FLAG == 0:
            if k < self.thres_l : 
                switch = 1
                self.FLAG = 1
            else : switch = 0  #stay
        elif self.FLAG == 1:
            if k > self.thres_h : 
                switch = 1
                self.FLAG = 0
            else : switch = 0  #stay
        
        print('FLAG:', self.FLAG)     
        
        return switch
        
        
#%% DQN : model-free RL


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
        self.fc3 = nn.Linear(32,2)
        
        
    def forward(self,x):
        
        x = torch.unsqueeze(x, -1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print('x:', x)
        return x
        
    def sample_action(self, obs, epsilon):
        
        obs = torch.from_numpy(np.array(int(obs)))
        out = self.forward(obs)
        if random.random() < epsilon:
            #print('random action!')
            return random.randint(0,1)
        else :
            return out.argmax().item()
        
class DQN:
    
    def __init__(self, env, batch_size=5, learning_rate=0.004, 
                 target_update_frequency=600, gamma=0.98):
        
        # Set environment
        self.env = env
        self.env.reset()
        
        self.state_size = 2
        #print('state size:', self.state_size)
        self.action_size = 2
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.q= Qnet()
        #second qnet with same structure and initial values, updated separately
        self.q_target = copy.deepcopy(self.q)

        self.target_update_frequency = target_update_frequency
        self.total_step_count = 0
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.q.parameters(), lr = learning_rate)
        self.loss_fn = F.smooth_l1_loss
        
        #self.controller = Controller()   # call in main
        self.transfer_info = None
        
        
    def predict(self, state):
        # predict action of state based on policy
        return self.env.action_space.sample()

    def optimize_model(self):

        for i in range(10):
            
            # Sample a minibatch of transitions from experience replay (list)
            transitions = self.memory.sample(self.batch_size)
            #print("sample batch")

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
            target_q = r + self.gamma * max_q_s_prime * done_mask
            #print(q_s_a.size(), target_q.size())
            
            loss = self.loss_fn(q_s_a, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            #print('i:',i)
            #print('backward loss update!')
            self.optimizer.step()
            
        
    
    def learn(self, max_episode, controller):
        
        episode_record = []     # save list to draw plot
        last_100_game_reward = deque(maxlen=100)
        
        #win_num = 0 

        print("=" * 70)
        print("begin mf learn")
        reward_sum = 0
        try: 
            for episode in range(max_episode):
                #print("episode:",episode)
                
                done = False
                state = self.env.reset()
                step_count = 0
                
                #epsilon = max(0.01, 0.1 - 0.1*1*(episode/max_episode))
                epsilon = 1
                
                
                # start one episode
                while not done:
                    if episode > 100 : self.env.render(state, reward_sum, episode)            
                    
                    # Select epsilon-greedy action
                    action = self.q.sample_action(state, epsilon)
                    #action = self.predict(state)
                    
                    #if episode < 100:
                    #    action = random.randint(0,2)
                        
                    # Take a step with current state and selected action
                    next_state, reward, done = self.env.step(state, action)
                    reward_sum += reward
                    
                    done_mask = 0.0 if done else 1.0
                    
                    print('state:{}, action:{}, reward:{}, next_state:{}'.format(state, action, reward, next_state))
                    self.memory.put(state, action, reward, next_state, done_mask)
                    
                    # Controller checks for model switch in the beginning of every trial
                    if state == 0 :
                        print("call controller")
                        
                        #self.controller.put(action, next_state)
                        controller.put(action, next_state)
                        #switch = self.controller.switch_model(thres_l=0.08, thres_h=0.1)
                        switch = controller.switch_model()
                        
                        if switch == 1:
                            print("SWITCH to model-based!")
                            # save info to transfer
                            self.transfer_info = self.memory   # class ReplayBuffer
                            raise StopIteration
                            
                        
                    #print('memory_size:' , self.memory.size())
                    
                    state = next_state
                    step_count += 1
                    self.total_step_count += 1
                    
                    #print('TOTAL_STEP_COUNT:',self.total_step_count)
    
                # Record the average reward of recent 100 episodes
                last_100_game_reward.append(reward_sum)
                avg_reward = np.mean(last_100_game_reward)
                episode_record.append(avg_reward)
                print("[Episode {:>5}]  episode steps: {:>5} avg: {}".format(episode, step_count, avg_reward))
                
                
                if self.memory.size() > 2000:
                        self.optimize_model()
                
                if self.total_step_count % self.target_update_frequency == 0 : 
                    print('++++++++++copy weight from target network+++++++++++++')
                    self.q_target.load_state_dict(self.q.state_dict())
        
        except StopIteration : pass 

        # pygame.quit()
        # sys.exit()
           
        #print('win_num:', win_num)              
        return episode_record, self.transfer_info, episode

