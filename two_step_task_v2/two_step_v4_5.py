
import numpy as np
from random import random
import pygame

#pygame.init()
#clock = pygame.time.Clock()
#set_color = (255,255,255)
#screen = pygame.display.set_mode((400, 500))
#pygame.display.set_caption("two step task")



#%% Define Environment Class %%

class Env():
    
    print('======imported Env============')
    
    is_running = True
    pygame.init()
    
    
    # Start
    
    total_reward = 0
    trial_num = 0 
    
    
    def __init__(self, rew_gen = 'blocks', block_length = 100, n_trials=1000):     # FIXME : change probs to 4 choices
        
        self.curr_state = 0
        self.reward = 0
        self.n_trials = n_trials   # Total number of trials
        self.rew_gen = rew_gen    # Which reward generator to use. : blocks, gaussian random walk...
        self.block_length = block_length # Length of each block.               
       
        self.probs = [[0.9, 0.1, 0.6, 0.4], [0.1, 0.9, 0.4, 0.6],[0.3, 0.8, 0.7, 0.3],[0.6, 0.4, 0.2, 0.8]]    #reward probabilities
        self.com_probs = [[0.85,0.15], [0.15,0.85]]    #transition probabilities
        
    def rule_changer(self):
        
        
        print("=============initiate rule=================")
        
        # Transition probabilities : random sequence of blocks
        rand_index_list = np.arange(2)
        np.random.shuffle(rand_index_list)
        
        block_1_trans = np.tile(self.com_probs[rand_index_list[0]],(self.block_length*2,1))
        block_2_trans = np.tile(self.com_probs[rand_index_list[1]],(self.block_length*2,1))

        self.trans_rule = np.tile(np.vstack([block_1_trans,block_2_trans]), 
                             (np.ceil((self.n_trials/(self.block_length*2*2))).astype('int'),1))[:self.n_trials,:]     

        # Reward probabilities : random sequence of blocks
        rand_index_list = np.arange(4)
        np.random.shuffle(rand_index_list)
        
        block_1_reward = np.tile(self.probs[rand_index_list[0]],(self.block_length,1))
        block_2_reward = np.tile(self.probs[rand_index_list[1]],(self.block_length,1))
        block_3_reward = np.tile(self.probs[rand_index_list[2]],(self.block_length,1))
        block_4_reward = np.tile(self.probs[rand_index_list[3]],(self.block_length,1))
        
        self.rew_rule = np.tile(np.vstack([block_1_reward,block_2_reward, block_3_reward, block_4_reward]), 
                           (np.ceil((self.n_trials/(self.block_length*4))).astype('int'),1))[:self.n_trials,:]     

        self.trans_probs = self.trans_rule
        self.reward_probs  = self.rew_rule
        self.rew_prob_iter = iter(self.reward_probs)
        self.trans_prob_iter = iter(self.trans_probs)
        
        print('trans_rule',self.trans_rule)
        print('rew_rule',self.rew_rule)
    
    def reset(self):
        
        self.curr_state = 0
        #next_state = 0
        self.reward = 0     
        self.trans_prob_iter = iter(self.trans_probs)
        
        return self.curr_state
        
    def step(self, curr_state, action):
        
        # based on our current state :
        # 0 = stage1, 1 = stage2A, 2 = right in stage2B
        # 3, 4, 5, 6 = results
        

        reward = 0 
        done = False 
        
        # stage 1
        if curr_state == 0:   
            transition  = int(random() < next(self.trans_prob_iter)[0])   # 1 if common, 0 if rare.
            next_state = int(action == transition) + 1    # based on probability, decide whether to follow action
            print('step - curr_state=0, nextstate is :', next_state)
            
        # stage 2
        elif curr_state == 1:  # 2A (left)
            reward = int(random() < next(self.rew_prob_iter)[curr_state-1 + action])
            done = True
            next_state= action + 3    # to save info of action in a certain state 
            
        elif curr_state == 2:  # 2B (right)
            reward = int(random() < next(self.rew_prob_iter)[curr_state + action])
            done = True
            next_state= action + 3
            
        return next_state, reward, done   # to save info of action in a certain state  
    
    

    def render(self,curr_state, total_reward, trial_num):
        
        print('render!')
        
        white = (255, 255, 255)
        green = (0, 255, 0)
        blue = (0, 0, 128)
        pygame.init()
        
        
            
        clock = pygame.time.Clock()
        pygame.display.set_caption('Two-step-tast-v2')
        screen = pygame.display.set_mode((800, 600))
        #background = pygame.Surface((800, 600))
        screen.fill(pygame.Color('#000000'))  # white
        clock.tick(30)
        
        if curr_state == -1:
            img_1 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\start.jpg")
            img_2 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\null.png")
        elif curr_state == 0:
            img_1 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\cat.png")
            img_2 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\cat_2.png")
        elif curr_state == 1:
            img_1 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\dog.jpg")
            img_2 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\dog_2.jpg")
        elif curr_state == 2:
            img_1 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\hamster.jpg")
            img_2 = pygame.image.load("C:\\Users\\Naeun Oh\\Downloads\\two_step_task_v2\\hamster_2.png")
        
        img_1 = pygame.transform.scale(img_1,(200,200))
        img_2 = pygame.transform.scale(img_2,(200,200))    
        #printGui()
        
        font = pygame.font.Font('freesansbold.ttf', 32)
        
        info = "curr_state:" + str(curr_state)
        reward_info = "total_reward:" + str(total_reward)
        trial_info = "trial:"+ str(trial_num)
        text = font.render(info, True, green, blue)
        text1 = font.render(reward_info, True, green, blue)
        text2 = font.render(trial_info, True, green, blue)


        screen.blit(img_1, (200, 200))
        screen.blit(img_2, (400, 200))
        screen.blit(text, (300,100))
        screen.blit(text1, (300,450))
        screen.blit(text2, (350,50))

        pygame.display.update()
        
        
    
    
    
    

    
    
    

        
        
        


    
    
    
    
    
        
        
        