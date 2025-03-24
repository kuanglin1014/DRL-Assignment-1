# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

is_pas = (True, True, True, True)
is_des = (True, True, True, True)
is_pickup = False
pas_loc = (0,0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def get_state(obs, is_pas, is_des, is_pickup, pas_loc):
    #TODO
    def get_vector(from_pos, to_pos):
        return (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
    def dis(from_pos, to_pos):
        return abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
    pas = list(is_pas)
    des = list(is_des)
    if pas[0] and (not dis((obs[2], obs[3]), (obs[0], obs[1])) <= 1 and obs[14]) or (dis((obs[2], obs[3]), (obs[0], obs[1])) <= 1 and not obs[14]):
        pas[0] = False
    if pas[1] and (not dis((obs[4], obs[5]), (obs[0], obs[1])) <= 1 and obs[14]) or (dis((obs[4], obs[5]), (obs[0], obs[1])) <= 1 and not obs[14]):
        pas[1] = False
    if pas[2] and (not dis((obs[6], obs[7]), (obs[0], obs[1])) <= 1 and obs[14]) or (dis((obs[6], obs[7]), (obs[0], obs[1])) <= 1 and not obs[14]):
        pas[2] = False
    if pas[3] and (not dis((obs[8], obs[9]), (obs[0], obs[1])) <= 1 and obs[14]) or (dis((obs[8], obs[9]), (obs[0], obs[1])) <= 1 and not obs[14]):
        pas[3] = False
    if (not dis((obs[2], obs[3]), (obs[0], obs[1])) <= 1 and obs[15]) or (dis((obs[2], obs[3]), (obs[0], obs[1])) <= 1 and not obs[15]):
        des[0] = False
    if (not dis((obs[4], obs[5]), (obs[0], obs[1])) <= 1 and obs[15]) or (dis((obs[4], obs[5]), (obs[0], obs[1])) <= 1 and not obs[15]):
        des[1] = False
    if (not dis((obs[6], obs[7]), (obs[0], obs[1])) <= 1 and obs[15]) or (dis((obs[6], obs[7]), (obs[0], obs[1])) <= 1 and not obs[15]):
        des[2] = False
    if (not dis((obs[8], obs[9]), (obs[0], obs[1])) <= 1 and obs[15]) or (dis((obs[8], obs[9]), (obs[0], obs[1])) <= 1 and not obs[15]):
        des[3] = False
    pas = tuple(pas)
    des = tuple(des)
    if not is_pickup:
        if pas[0]:
            return (get_vector((obs[2], obs[3]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
        if pas[1]:
            return (get_vector((obs[4], obs[5]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
        if pas[2]:
            return (get_vector((obs[6], obs[7]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
        if pas[3]:
            return (get_vector((obs[8], obs[9]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
    else:
        if des[0]:
            return (get_vector((obs[2], obs[3]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
        if des[1]:
            return (get_vector((obs[4], obs[5]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
        if des[2]:
            return (get_vector((obs[6], obs[7]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
        if des[3]:
            return (get_vector((obs[8], obs[9]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)
    return (get_vector((pas_loc[0], pas_loc[1]), (obs[0], obs[1])), obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], pas, des)

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    #return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

    global is_pas
    global is_des
    global is_pickup
    global pas_loc
    state = get_state(obs, is_pas, is_des, is_pickup, pas_loc) + (is_pickup,)
    is_pas = state[7]
    is_des = state[8]
    action = None
    if state not in q_table:
        #print(obs)
        action = random.choice([0, 1, 2, 3])
    else:
        if sum(is_pas) > 1:
            action_probs = q_table[state][:4]
            action = np.random.choice(4, p=softmax(action_probs))
        else:
            action_probs = q_table[state]
            action = np.random.choice(6, p=softmax(action_probs))
        if action == 4 and obs[14]:
            is_pickup = True
        if action == 5 and is_pickup:
            is_pickup = False
            pas_loc = (obs[0], obs[1])
    #print('**',action,obs)
    return action
