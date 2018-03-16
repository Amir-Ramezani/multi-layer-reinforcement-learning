######

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.models import load_model
from keras.models import model_from_json

import numpy as np
import random

##################################### neural network function
global reply_memory_inputs, reply_memory_outputs, reply_memory_index, reply_memory_counter, Reply_Memory_On, Reply_Memory_New_On, reply_memory_previous_state, reply_memory_previous_action,reply_memory_current_state, reply_memory_reward,reply_memory_outputs_updated

'''
reply_memory_inputs=np.zeros((32,400), dtype=np.float32) # memroy based policy, for ultrasonics
reply_memory_outputs=np.zeros((32,3), dtype=np.float32) #five actions and one action value

reply_memory_previous_states=np.zeros((50000,400), dtype=np.float32) # memroy based policy, for 3 ultrasonic
reply_memory_previous_actions=np.zeros((50000,1), dtype=np.int16)
reply_memory_current_states=np.zeros((50000,400), dtype=np.float32) #memory based policy
reply_memory_rewards=np.zeros((50000,1), dtype=np.int16)
reply_memory_q_values=np.zeros((50000,3), dtype=np.float32) #five actions and one action value

selected_reply_memory_previous_states=np.zeros((32,400), dtype=np.float32) # memroy based policy, for 3 ultrasonic
selected_reply_memory_previous_actions=np.zeros((32,1), dtype=np.int16)
selected_reply_memory_current_states=np.zeros((32,400), dtype=np.float32) #memory based policy
selected_reply_memory_rewards=np.zeros((32,1), dtype=np.int16)
selected_reply_memory_q_values=np.zeros((32,3), dtype=np.float32) #five actions and one action value
'''

#changing the types of the memory replay because of the memory from float byte
reply_memory_inputs=np.zeros((32,640), dtype=np.float32) # memroy based policy, for ultrasonics
reply_memory_outputs=np.zeros((32,3), dtype=np.float32) #five actions and one action value

reply_memory_previous_states=np.zeros((50000,640), dtype=np.float32) # memroy based policy, for 3 ultrasonic
reply_memory_previous_actions=np.zeros((50000,1), dtype=np.uint8)
reply_memory_current_states=np.zeros((50000,640), dtype=np.float32) #memory based policy
reply_memory_rewards=np.zeros((50000,1), dtype=np.int16)
reply_memory_q_values=np.zeros((50000,3), dtype=np.float16) #five actions and one action value

selected_reply_memory_previous_states=np.zeros((32,640), dtype=np.float32) # memroy based policy, for 3 ultrasonic
selected_reply_memory_previous_actions=np.zeros((32,1), dtype=np.uint8)
selected_reply_memory_current_states=np.zeros((32,640), dtype=np.float32) #memory based policy
selected_reply_memory_rewards=np.zeros((32,1), dtype=np.int16)
selected_reply_memory_q_values=np.zeros((32,3), dtype=np.float16) #five actions and one action value

reply_memory_index=0
reply_memory_counter=0

def train_3actions_DepthP1(model, state, state_action_value_list, action, current_state, reward, epsilon, Reply_Memory_On, Reply_Memory_New_On, Reply_Memory_New_On_Update,Alpha,Lambda, Gamma):
    global reply_memory_inputs, reply_memory_outputs, reply_memory_index, reply_memory_counter, reply_memory_previous_state, reply_memory_previous_action, reply_memory_current_state, reply_memory_reward, reply_memory_outputs_updated

    state_action_value_list=state_action_value_list

    #print(state)
    #print(action)
    #print(state_action_value)
    #print("Reply Memory On: ", Reply_Memory_On)
    #print("Reply Memory New On: ", Reply_Memory_New_On)
    #Reply_Memory_New_On=False

    ### regarding the reply memory

    if (Reply_Memory_On == False):

        #model.fit(reply_memory_inputs[reply_memory_index, :].reshape(1, 4, 60, 80),
        #          reply_memory_outputs[reply_memory_index, :].reshape(1, 3), nb_epoch=50,
        #          batch_size=1, verbose=0)
        model.train_on_batch(reply_memory_inputs[reply_memory_index, :].reshape(1, 640),
                  reply_memory_outputs[reply_memory_index, :].reshape(1, 3))

    elif(Reply_Memory_On==True and Reply_Memory_New_On==False):
        #print("Reply Memory")
        if (reply_memory_counter < 32):
            reply_memory_counter = reply_memory_counter + 1

        reply_memory_inputs[reply_memory_index] = state.reshape(640)
        reply_memory_outputs[reply_memory_index]=state_action_value_list

        #model.fit(reply_memory_inputs[0:reply_memory_counter, :].reshape(reply_memory_counter,4, 60, 80),
        #          reply_memory_outputs[0:reply_memory_counter, :].reshape(reply_memory_counter, 3), nb_epoch=50,
        #          batch_size=reply_memory_counter, verbose=0)

        model.train_on_batch(reply_memory_inputs[0:reply_memory_counter, :].reshape(reply_memory_counter,640),
                  reply_memory_outputs[0:reply_memory_counter, :].reshape(reply_memory_counter, 3))

        reply_memory_index = reply_memory_index + 1

        if (reply_memory_index > 31):
            reply_memory_index = 0

        #print("State in reply memory: ", reply_memory_inputs[reply_memory_index])
        #print("State in reply memory: ", reply_memory_outputs[reply_memory_index])

    elif (Reply_Memory_On==True and Reply_Memory_New_On == True):
        print "Reply Memory New",

        if (reply_memory_counter < 50000):
            reply_memory_counter = reply_memory_counter + 1

        #extra section for memory reply
        #we try to update the values with the correct one before updating

        reply_memory_previous_states[reply_memory_index]=state.reshape(640)
        reply_memory_previous_actions[reply_memory_index] = action
        reply_memory_current_states[reply_memory_index] = current_state.reshape(640)
        reply_memory_rewards[reply_memory_index] = reward
        reply_memory_q_values[reply_memory_index] = state_action_value_list

        if(reply_memory_counter>31):
            #random_numbers = random.sample(range(0, 50000), 31)

            random_numbers = random.sample(range(0, reply_memory_counter), 31)

            for i in range(0, 31):
                #print("Random Number: %s" %random_numbers[i])

                selected_reply_memory_previous_states[i] = reply_memory_previous_states[random_numbers[i]].reshape(640)
                selected_reply_memory_previous_actions[i] = reply_memory_previous_actions[random_numbers[i]]
                selected_reply_memory_rewards[i] = reply_memory_rewards[random_numbers[i]]
                selected_reply_memory_current_states[i] = reply_memory_current_states[random_numbers[i]].reshape(640)
                selected_reply_memory_q_values[i] = reply_memory_q_values[random_numbers[i]]

        elif(reply_memory_counter<31):
            i=0
            counter=0
            while(i<31):
                #print("counter: %s" %counter)
                #print("I: %s" %i)

                selected_reply_memory_previous_states[i] = reply_memory_previous_states[counter].reshape(640)
                selected_reply_memory_previous_actions[i] = reply_memory_previous_actions[counter]
                selected_reply_memory_rewards[i] = reply_memory_rewards[counter]
                selected_reply_memory_current_states[i] = reply_memory_current_states[counter].reshape(640)
                selected_reply_memory_q_values[i] = reply_memory_q_values[counter]

                counter = counter + 1
                if(counter>=reply_memory_counter):
                    counter=0

                i=i+1

        if(Reply_Memory_New_On_Update==True):
            for r in range(0,31):

                #randomly selecting among
                #i=0
                #while(i<31):
                #    random_number = np.random.randint(0, 320)  # we have three actions so we choose between them

                #updating values

                #Initialization

                previous_state = selected_reply_memory_previous_states[r].reshape(640)
                previous_action=selected_reply_memory_previous_actions[r]
                reward=selected_reply_memory_rewards[r]
                current_state = selected_reply_memory_current_states[r].reshape(640)

                #print("Previous State: ")
                #print(previous_state)
                #print("Previous Action: ")
                #print(previous_action)
                #print("Current State: ")
                #print(current_state)
                #print("Reward: ")
                #print(reward)

                previous_action_values=predict_3actions_DepthP1(model,previous_state)
                current_action_values = predict_3actions_DepthP1(model, current_state)

                new_action=0
                QLearning=True
                if(QLearning==True):
                    ###### Q Learning
                    new_action = (np.argmax(current_action_values))

                else: ###### SARSA
                    if (random.random() < epsilon):  # choose random action (Random==True):#
                        new_action = np.random.randint(0, 3)  # we have three actions so we choose between them

                    else:  # choose best action from Q(s,a) values
                        new_action = (np.argmax(current_action_values))

                new_action_value = current_action_values[0][new_action]

                previous_action_value = previous_action_values[0][previous_action]

                previous_action_new_value=previous_action_value+Alpha*(reward + Gamma*(new_action_value)-previous_action_value)

                previous_action_values[0][previous_action]=previous_action_new_value

                selected_reply_memory_q_values[r] = previous_action_values#multiplication by 2 is because of dropout

                #print("X-Previous Action Values: %s" %previous_action_values)


        #add all the state for update
        reply_memory_inputs[:]=selected_reply_memory_previous_states[:]
        reply_memory_outputs[:]=selected_reply_memory_q_values[:]


        # add the current state
        reply_memory_inputs[31]=state
        reply_memory_outputs[31]=state_action_value_list


        #print("Reply Memory Outputs: %s" %reply_memory_outputs)

        #model.fit(reply_memory_inputs[0:100, :].reshape(100,4, 60, 80),
        #          reply_memory_outputs[0:100, :].reshape(100, 3), nb_epoch=100,
        #          batch_size=100, verbose=0)

        model.train_on_batch(reply_memory_inputs[0:32, :].reshape(32,640),
                  reply_memory_outputs[0:32, :].reshape(32, 3))

        reply_memory_index = reply_memory_index + 1

        if (reply_memory_index > 49999):
            reply_memory_index = 0




    return model

def predict_3actions_DepthP1(model, state):
    return model.predict(state.reshape(1,640), batch_size=1)


def create_model():
    print("creating model...")
    print("K image dim ordering: %s" % (K.image_dim_ordering(),))

    img_rows = 60
    img_cols = 80

    if K.image_dim_ordering() == 'th':
        input_shape = (4, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 4)

    model = Sequential()

    # this was good for normal reward for three sensor
    #model.add(Convolution2D(32, 8, 8, border_mode='valid', subsample=(4, 4), input_shape=input_shape))
    #model.add(Activation('relu'))
    ##model.add(MaxPooling2D(pool_size=(2, 2)))
    ## model.add(Dropout(0.5))


    #model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    #model.add(Activation('relu'))
    ##model.add(MaxPooling2D(pool_size=(2, 2)))
    ## model.add(Dropout(0.5))

    #model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    #model.add(Activation('relu'))
    ##model.add(MaxPooling2D(pool_size=(2, 2)))
    ## model.add(Dropout(0.5))


    ##model.add(Convolution2D(64, 3, 3))
    ##model.add(Activation('relu'))
    ##model.add(MaxPooling2D(pool_size=(2, 2)))
    ## model.add(Dropout(0.5))

    ## model.add(Dense(50, input_shape=(80,60), init='zero', activation='linear'))
    ## model.add(Dropout(0.5))

    #model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(512, input_dim=640, init='normal', activation='relu'))

    model.add(Dense(256, init='normal', activation='relu'))
    #model.add(Dense(1000, init='normal', activation='relu'))

    # model.add(Dropout(0.5))
    model.add(Dense(3, init='normal', activation='linear'))
    # model.add(Dropout(0.5))

    '''
    # this was good for normal reward for three sensor
    model.add(Convolution2D(16, 8, 8, border_mode='valid', subsample=(1, 1), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))


    model.add(Convolution2D(32, 4, 4, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))


    #model.add(Convolution2D(64, 3, 3))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    # model.add(Dense(50, input_shape=(80,60), init='zero', activation='linear'))
    # model.add(Dropout(0.5))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(256, init='normal', activation='relu'))
    #model.add(Dense(1000, init='normal', activation='relu'))

    # model.add(Dropout(0.5))
    model.add(Dense(3, init='normal', activation='linear'))
    # model.add(Dropout(0.5))
    #
    '''

    # print(model.summary())
    # print(model.optimizer[0])

    #adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # learning rate (default) = 0.002

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # model.compile(loss='mse', optimizer=adamax)
    model.compile(loss='mse', optimizer=rms)

    print("model created...X")

    return model

def load_model(filename_str):
    print("load and train the model...")

    # load json and create model
    json_file = open(filename_str + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(filename_str + ".h5")

    adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                    decay=0.0)  # learning rate (default) = 0.002

    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(loss='mse', optimizer=rms)

    print("Loaded model from disk")

    return model