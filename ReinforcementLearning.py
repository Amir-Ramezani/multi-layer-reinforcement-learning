#######

import numpy as np
import random

from NeuralNetwork import train_3actions_DepthP1, predict_3actions_DepthP1
from KobukiFunctions import move_turtle_bot

################################## reinfocement learning functions

# mechanism for n-step=Lambda reinforcement learning
global Z_S_A, Z_S_A_index, Z_S_A_counter, N_Step_On, Z_S_A_inputs, Z_S_A_outputs

Z_S_A=np.zeros((100,19), dtype=np.float32) # for memeory based

Z_S_A_inputs=np.zeros((100,18), dtype=np.float32) # for memeory based

Z_S_A_outputs=np.zeros((100,1), dtype=np.float32) #five actions
Z_S_A_index=0
Z_S_A_counter=0


############################## Normal Q-Learning and SARSA - Artificial Neural Network - Using Depth Image - Phase 1
def get_reward_ANN_DepthP1(action, previous_state, current_state):
    accumulative_reward=0


    if(current_state[0]==1):
        accumulative_reward = accumulative_reward-150

    if(current_state[3]==1): #sensor center
        accumulative_reward = accumulative_reward-200

    if(current_state[6]==1):
        accumulative_reward = accumulative_reward-150

    if(action==0):
        if(current_state[0] == 0 and current_state[3] == 0 and current_state[6] == 0): # this is one change
            accumulative_reward = accumulative_reward + 50

    if (action==1 or action==2 or action == 3 or action == 4):
        accumulative_reward = accumulative_reward - 25

    return accumulative_reward

def make_move_ANN_DepthP1(current_state,model,epsilon, dC, dL, dR):
    print("e=%s," % (epsilon,)),

    state_action_value_list=predict_3actions_DepthP1(model,current_state)

    #epsilon-Greedy policy
    if(random.random() < epsilon):  # choose random action (Random==True):#
        print "Act=Random,",
        action = np.random.randint(0, 3)  # we have three actions so we choose between them

    else:  # choose best action from Q(s,a) values
        print "Act=Policy,",
        possible_actions=state_action_value_list
        action = (np.argmax(possible_actions))

    move_turtle_bot(action,dC,dL,dR)

    return action, state_action_value_list

def update_Q_value_ANN_DepthP1(previous_state,previous_action, current_state,reward,model,epsilon, N_Step_On,Alpha,Lambda, Gamma, Reply_Memory_On, Reply_Memory_New_On, Reply_Memory_New_On_Update):
    global Z_S_A, Z_S_A_index, Z_S_A_counter,  Z_S_A_inputs, Z_S_A_outputs
    QLearning=True
    #print("Alpha (Update Function): ", Alpha)
    #print("Gamma (Update Function): ", Gamma)
    #print("Lambda (Update Function): ", Lambda)

    previous_possible_actions=predict_3actions_DepthP1(model,previous_state)
    possible_actions = predict_3actions_DepthP1(model, current_state)

    #print(previous_state)
    #print(current_state)

    #print(previous_possible_actions)
    #print(possible_actions)

    new_action=0

    if(QLearning==True):
        print "Q-Learning,",
        ###### Q Learning
        new_action = (np.argmax(possible_actions))

    else:
        print "SARSA,",
        ###### SARSA
        # epsilon-Greedy policy
        if (random.random() < epsilon):  # choose random action (Random==True):#
            print "Act2=Random,",
            new_action = np.random.randint(0, 3)  # we have three actions so we choose between them

        else:  # choose best action from Q(s,a) values
            print "Act2=Policy,",
            new_action = (np.argmax(possible_actions))

    new_action_value = possible_actions[0][new_action]

    print "Before: %.2f,%.2f,%.2f," % (previous_possible_actions[0][0],previous_possible_actions[0][1],previous_possible_actions[0][2]),#,previous_possible_actions[3],previous_possible_actions[4],),


    previous_action_value = previous_possible_actions[0][previous_action]
    #print(previous_action_value)
    if(N_Step_On==False):
        ### Regarding the 1-Step TD Learning

        previous_action_new_QValue=previous_action_value+Alpha*(reward + Gamma*(new_action_value)-previous_action_value)

        print ("New Q-Value: %.2f," % (previous_action_new_QValue,)),

        previous_possible_actions[0][previous_action]=previous_action_new_QValue

        #print("Exactly before training...")
        train_3actions_DepthP1(model, previous_state, previous_possible_actions, previous_action, current_state, reward,
                               epsilon, Reply_Memory_On, Reply_Memory_New_On, Reply_Memory_New_On_Update,Alpha,Lambda, Gamma)

    else:
        print "4Steps, TD(Lambda)",

    ############################################################

    previous_possible_actions = predict_3actions_DepthP1(model, previous_state)

    print "After: %.2f,%.2f,%.2f," % (previous_possible_actions[0][0],previous_possible_actions[0][1],previous_possible_actions[0][2]),#,previous_possible_actions[3],previous_possible_actions[4],),
