# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_state(obs):
    #TODO
    return (obs[10], obs[11], obs[12], obs[13])

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    #return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

    state = get_state(obs)
    action = None
    if state not in q_table:
        #print(obs)
        action = random.choice([0, 1, 2, 3])
    else:
        action = np.argmax(q_table[state][:4])
    #print('**',action,obs)
    return action
