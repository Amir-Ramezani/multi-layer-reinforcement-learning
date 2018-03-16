## more to ros

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Range
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import MotorPower

## more to neural networks

import matplotlib.pyplot as plt


import numpy as np
import random
import pickle
import theano
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


## more to opencv

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tempfile import TemporaryFile
########################################

rotateLeft=0
rotateRight=0

global sensors

sensors=np.zeros((5))

sensors[0]=150 #left sensor
sensors[1]=150  #center sensor
sensors[2]=150 #right sensor
sensors[3]=150 #left 5 sensor
sensors[4]=150 #right 5 sensor

global sensor_DP1, image_DP1

image_DP1 = np.zeros((1, 80, 60), np.float32)

sensor_DP1 = np.zeros((1, 80), np.float32)

global weight_matrix
weight_matrix = np.zeros((256, 80, 1), np.double)

global weighted_reward, previous_state_weighted_reward, current_state_weighted_reward
weighted_reward = np.zeros((1, 80), np.int32)
previous_state_weighted_reward = np.zeros((1, 80), np.int32)
current_state_weighted_reward = np.zeros((1, 80), np.int32)

obstacle=True
obstacle_avoided=True
publish=False

global turtlebot_state

turtlebot_state=[]

########################################

def init():
    rospy.init_node('kobuki_control', anonymous=True)
    global velocity_publisher
    velocity_publisher= rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
    global cmd
    cmd=Twist()
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.0

    ##########################################

    #Speed of movements of the ROBOT
    global ros_sleep_time
    #ros_sleep_time=1.0 #0.6

    ros_sleep_time = 1.0  # 0.6

    #learning rate of the neural network
    global neural_net_learning_rate
    neural_net_learning_rate=0.01

    #load a model or no
    global load_model_scenario
    load_model_scenario=False

    ###########################################

    velocity_publisher.publish(cmd)
    rospy.sleep(ros_sleep_time)

    init_turtlebot_state()

    return True

#Define States
def init_turtlebot_state():
    global previous_state,new_state,current_state,number_of_states,number_of_action_values,action,action_value,Alpha,Lambda, Gamma, previous_state_DP1, current_state_DP1

    previous_state_DP1 = np.zeros((1, 240), np.float32) #memory based policy

    #previous_state_DP1 = np.zeros((1, 80), np.float32)

    current_state_DP1 = np.zeros((1, 240), np.float32) #memory based policy

    #current_state_DP1 = np.zeros((1, 80), np.float32)

    previous_state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    #global
    new_state = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)

    #global
    current_state = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)

    #global
    number_of_states = 0

    #global
    number_of_action_values = 0

    #global
    action = 0

    #global
    action_value = 0.0

    #################################################################
    #global
    Alpha = 0.5 # 0.1 # learning rate // it was 0.5

    #global
    Gamma = 0.1 # effect of the future reward

    #global
    Lambda= 0.2 # n-step learning
    #################################################################


    for sensorLeft0 in range(0,2):
        for sensorLeft1 in range(0,2):
            for sensorLeft2 in range(0,2):
                 for sensorCenter0 in range(0,2):
                     for sensorCenter1 in range(0,2):
                        for sensorCenter2 in range(0,2):
                           for sensorRight0 in range(0,2):
                               for sensorRight1 in range(0,2):
                                   for sensorRight2 in range(0,2):
                                       for sensorLeft50 in range(0,2):
                                           for sensorLeft51 in range(0,2):
                                               for sensorRight50 in range(0,2):
                                                   for sensorRight51 in range(0,2):
                                                       turtlebot_state.append((sensorLeft0,sensorLeft1,sensorLeft2,sensorCenter0,sensorCenter1,sensorCenter2,sensorRight0,sensorRight1,sensorRight2))
                                            #turtlebot_state.append((sensorLeft0, sensorLeft1, sensorCenter0, sensorCenter1,
                                                                 #sensorRight0, sensorRight1))
                                                       number_of_states=number_of_states+1

    global turtlebot_state_action_value
    turtlebot_state_action_value = {}

    for state in turtlebot_state:
        turtlebot_state_action_value[(state,0)]=0.0
        turtlebot_state_action_value[(state,1)]=0.0
        turtlebot_state_action_value[(state,2)]=0.0

        number_of_action_values=number_of_action_values+3



    return 0

def save_obj(obj, name ): # saving the files
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ): # loading the files
    with open(name, 'rb') as f:
        return pickle.load(f)

def normalize_data(actions_value):
    max_value=actions_value[(np.argmax(actions_value))][0]
    min_value = actions_value[(np.argmin(actions_value))][0]

    for item in actions_value:
        item[0] = ((item[0]) - (min_value)) / (max_value - min_value)

    return actions_value

def chatterCallback_sonarSensorL1(msg):
    global distanceLeft, distanceCenter, distanceRight
    distanceLeft=int((msg.ranges[0])*100)

def chatterCallback_sonarSensor(msg):
    global distanceLeft, distanceCenter, distanceRight
    distanceCenter=int((msg.ranges[0])*100)

def chatterCallback_sonarSensorR1(msg):
    global distanceLeft, distanceCenter, distanceRight, distanceLeft5, distanceRight5
    distanceRight=int((msg.ranges[0])*100)
    sensors[0]= distanceLeft
    sensors[1]= distanceCenter
    sensors[2]= distanceRight
    sensors[3] = distanceLeft5
    sensors[4] = distanceRight5
    #print(sensors)

def chatterCallback_sonarSensorR5(msg):
    global distanceLeft, distanceCenter, distanceRight
    distanceRight=int((msg.ranges[0])*100)
    #print(sensors)

def chatterCallback_sonarSensorL5(msg):
    global distanceLeft, distanceCenter, distanceRight, distanceLeft5, distanceRight5
    distanceLeft5=int((msg.ranges[0])*100)

def chatterCallback_sonarSensorR5(msg):
    global distanceLeft, distanceCenter, distanceRight, distanceLeft5, distanceRight5
    distanceRight5 = int((msg.ranges[0]) * 100)

def create_weighting_matrix():
    global weight_matrix

    weight_matrix = np.zeros((256, 160, 1), np.double)

    for i in range(0,160):
        for j in range(0,255):
            half=160/2
            if(i>half):
                val=half-(i-half)
            else:
                val=i

            val=val-(j/5)

            if(val<0):
                val=0

            weight_matrix[j,i]=val

    #return weight_matrix

def create_weighting_matrix_240():
    global weight_matrix

    weight_matrix = np.zeros((256, 240, 1), np.double)

    for i in range(0,240):
        for j in range(0,255):
            half=240/2
            if(i>half):
                val=half-(i-half)
            else:
                val=i

            val=val-(j/5)

            if(val<0):
                val=0

            weight_matrix[j,i]=val

    #return weight_matrix

def image_callback(data):
    global sensor_DP1,image_DP1

    global weight_matrix, weighted_reward

    global distanceLeft, distanceCenter, distanceRight, distanceLeft5, distanceRight5

    max_depth=0

    try:
        cv_image = bridge.imgmsg_to_cv2(data)  # 32FC1
        cv_image = cv_image
        #cv_image_to_show = np.float32(cv_image)/5
        cv_image_to_show = cv_image / 5
        #cv2.imwrite("/home/amir-ai/image1.jpg", cv_image)
        #for i in range(0, 240):
            #print(cv_image[i,320])
        #max_depth=np.nanmax(cv_image)
        #print(max_depth)

    except CvBridgeError as e:
        print("error here")
        print(e)

    #blank_image = np.zeros((256, 640, 1), np.uint8)

    cv_image_resized = cv2.resize(cv_image, (80, 60), interpolation=cv2.INTER_AREA)



    # min = np.nanmin(cv_image_resized[20:30, i])

    # min=min/5.5*255

    # if(min>255):
    #    min = 255
    cv_image_resized[1, :] = 255#(float)(distanceLeft + 10) / 100

    image_size=80
    for i in range(0, image_size):
        if (i < 20):
            if(distanceLeft < 40):
                cv_image_resized[:,i] = (float)(distanceLeft + 5)/100
            else:
                cv_image_resized[np.isnan(cv_image_resized[:,i]),i]=5.0
        elif (i > 19 and i < 60):
            if(distanceCenter < 50):
                cv_image_resized[:,i] = (float)(distanceCenter + 5) / 100
            else:
                cv_image_resized[np.isnan(cv_image_resized[:,i]),i]=5.0
        elif (i > 59 and i < 80):
            if(distanceRight < 50):
                cv_image_resized[:,i] = (float)(distanceRight + 5) / 100
            else:
                cv_image_resized[np.isnan(cv_image_resized[:,i]),i]=5.0

    image_DP1=np.array(cv_image_resized)
    #print("inside: ")
    #print(image_DP1)


    #cv_image_resized = cv2.resize(cv_image, (160, 120), interpolation=cv2.INTER_AREA)

    # cv2.imwrite("/home/amir-ai/image1.jpg", cv_image)
    #for i in range(0, 120):
        #print(cv_image_resized[i,80])

    blank_image_resized = np.zeros((256, 80, 1), np.uint8)
    #blank_image_resized = np.zeros((256, 160, 1), np.uint8)
    #blank_image_resized = np.zeros((256, 240, 1), np.uint8)

    max = 0
    min = 0

    cv_image_resized[20, :] = 255
    cv_image_resized[21, :] = 255
    cv_image_resized[29, :] = 255
    cv_image_resized[30, :] = 255

    #cv_image_resized[40, :] = 255
    #cv_image_resized[41, :] = 255
    #cv_image_resized[59, :] = 255
    #cv_image_resized[60, :] = 255

    cv_image_to_show[220, :] = 255
    cv_image_to_show[221, :] = 255
    cv_image_to_show[259, :] = 255
    cv_image_to_show[260, :] = 255


    image_size=80

    for i in range(0, image_size):
    #for i in range(0, 240):
        if(image_size==240):
            if(i>39 and i<200):
                min = np.nanmin(cv_image_resized[40:60, i-40])

                min=min/5.5*255

                if(min>255):
                    min = 255



            if(i<40):
                if(distanceLeft5==100):
                    min=255
                else:
                    min = distanceLeft5
            elif (i > 39 and i < 80 and distanceLeft < 40):
                min=distanceLeft
                #min=20
            elif (i>79 and i < 160 and distanceCenter < 40):
                min=distanceCenter
                #min=20
            elif (i > 159 and i < 200 and distanceRight < 40):
                min=distanceRight
                #min=20
            elif (i > 199 and i < 240):
                if(distanceRight5==100):
                    min=255
                else:
                    min=distanceRight5

        elif(image_size==160):
            min = np.nanmin(cv_image_resized[40:60, i])

            min=min/5.5*255

            if(min>255):
                min = 255

            if (i < 40 and distanceLeft < 40):
                min=distanceLeft
                min=20
            elif (i>39 and i < 120 and distanceCenter < 40):
                min=distanceCenter
                #min=20
            elif (i > 119 and i < 160 and distanceRight < 40):
                min=distanceRight
                min=20

        elif(image_size==80):
            min = np.nanmin(cv_image_resized[20:30, i])

            min=min/5.5*255

            if(min>255):
                min = 255

            if (i < 20 and distanceLeft < 50):
                min=distanceLeft
                #min=20
            elif (i>19 and i < 60 and distanceCenter < 50):
                min=distanceCenter
                #min=20
            elif (i > 59 and i < 80 and distanceRight < 50):
                min=distanceRight
                #min=20


        try:
            min=int(min)

            blank_image_resized[min, i] = 255
            sensor_DP1[0, i] = min
            val=(int(weight_matrix[min,i])*255)/100
            weighted_reward[0,i]=val
        except:
            print(min)

    cv2.imshow("Blank Image Resized", blank_image_resized)
    cv2.imshow("Image Depth", cv_image_resized)
    cv2.waitKey(3)

def image_callback_RGB(data):
    global sensor_RGB_DP1

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")  # 32FC1
        #cv_image = np.float32(cv_image) / 5
    except CvBridgeError as e:
        print("error here")
        print(e)

    cv2.imshow("Image RGB", cv_image)
    cv2.waitKey(3)

##################################### neural network function

global reply_memory_inputs, reply_memory_outputs, reply_memory_index, reply_memory_counter, Reply_Memory_On, Reply_Memory_New_On, reply_memory_previous_state, reply_memory_previous_action,reply_memory_current_state, reply_memory_reward,reply_memory_outputs_updated

# mechanism for n-step=Lambda reinforcement learning
global Z_S_A, Z_S_A_index, Z_S_A_counter, N_Step_On, Z_S_A_inputs, Z_S_A_outputs

Z_S_A=np.zeros((100,19), dtype=np.float32) # for memeory based

Z_S_A_inputs=np.zeros((100,18), dtype=np.float32) # for memeory based

Z_S_A_outputs=np.zeros((100,1), dtype=np.float32) #five actions
Z_S_A_index=0
Z_S_A_counter=0

reply_memory_inputs=np.zeros((100,1,240), dtype=np.float32) # memroy based policy, for ultrasonics
reply_memory_outputs=np.zeros((100,3), dtype=np.float32) #five actions and one action value

reply_memory_previous_states=np.zeros((320,1,240), dtype=np.float32) # memroy based policy, for 3 ultrasonic
reply_memory_previous_actions=np.zeros((320,1), dtype=np.int16)
reply_memory_current_states=np.zeros((320,1,240), dtype=np.float32) #memory based policy
reply_memory_rewards=np.zeros((320,1), dtype=np.int16)
reply_memory_q_values=np.zeros((320,3), dtype=np.float32) #five actions and one action value

selected_reply_memory_previous_states=np.zeros((100,1,240), dtype=np.float32) # memroy based policy, for 3 ultrasonic
selected_reply_memory_previous_actions=np.zeros((100,1), dtype=np.int16)
selected_reply_memory_current_states=np.zeros((100,1,240), dtype=np.float32) #memory based policy
selected_reply_memory_rewards=np.zeros((100,1), dtype=np.int16)
selected_reply_memory_q_values=np.zeros((100,3), dtype=np.float32) #five actions and one action value


reply_memory_index=0
reply_memory_counter=0

def train_3actions_DepthP1(model, state, state_action_value_list, action, current_state, reward, epsilon):
    global QLearning,reply_memory_inputs, reply_memory_outputs, reply_memory_index, reply_memory_counter, Reply_Memory_On, Reply_Memory_New_On, reply_memory_previous_state, reply_memory_previous_action, reply_memory_current_state, reply_memory_reward, reply_memory_outputs_updated
    global Reply_Memory_New_On_Update

    state_action_value_list=state_action_value_list

    #print(state)
    #print(action)
    #print(state_action_value)
    #print("Reply Memory On: ", Reply_Memory_On)
    #print("Reply Memory New On: ", Reply_Memory_New_On)
    #Reply_Memory_New_On=False

    ### regarding the reply memory

    if (Reply_Memory_On == False):
        print "Reply Memory Off",


        model.fit(reply_memory_inputs[reply_memory_index, :].reshape(1, 18),
                  reply_memory_outputs[reply_memory_index, :].reshape(1, 1), nb_epoch=50,
                  batch_size=1, verbose=0)

    elif(Reply_Memory_On==True and Reply_Memory_New_On==False):
        print("Reply Memory")
        if (reply_memory_counter < 100):
            reply_memory_counter = reply_memory_counter + 1

        reply_memory_inputs[reply_memory_index] = state.reshape(1, 240)
        reply_memory_outputs[reply_memory_index]=state_action_value_list

        #print(reply_memory_inputs[0:reply_memory_counter])
        #print(reply_memory_outputs[0:reply_memory_counter])

        model.fit(reply_memory_inputs[0:reply_memory_counter, :].reshape(reply_memory_counter, 240),
                  reply_memory_outputs[0:reply_memory_counter, :].reshape(reply_memory_counter, 3),
                  nb_epoch=150, batch_size=reply_memory_counter, verbose=0)

        reply_memory_index = reply_memory_index + 1

        if (reply_memory_index > 99):
            reply_memory_index = 0

        #print("State in reply memory: ", reply_memory_inputs[reply_memory_index])
        #print("State in reply memory: ", reply_memory_outputs[reply_memory_index])

    elif (Reply_Memory_On==True and Reply_Memory_New_On == True):
        print "Reply Memory New",

        if (reply_memory_counter < 320):
            reply_memory_counter = reply_memory_counter + 1

        #extra section for memory reply
        #we try to update the values with the correct one before updating

        reply_memory_previous_states[reply_memory_index]=state.reshape(1,240)
        reply_memory_previous_actions[reply_memory_index] = action
        reply_memory_current_states[reply_memory_index] = current_state.reshape(1,240)
        reply_memory_rewards[reply_memory_index] = reward
        reply_memory_q_values[reply_memory_index] = state_action_value_list

        if(reply_memory_counter>99):

            random_numbers = random.sample(range(0, reply_memory_counter), 99)

            for i in range(0, 99):
                #print("Random Number: %s" %random_numbers[i])

                selected_reply_memory_previous_states[i] = reply_memory_previous_states[random_numbers[i]].reshape(1, 240)
                selected_reply_memory_previous_actions[i] = reply_memory_previous_actions[random_numbers[i]]
                selected_reply_memory_rewards[i] = reply_memory_rewards[random_numbers[i]]
                selected_reply_memory_current_states[i] = reply_memory_current_states[random_numbers[i]].reshape(1, 240)
                selected_reply_memory_q_values[i] = reply_memory_q_values[random_numbers[i]]

        elif(reply_memory_counter<99):
            i=0
            counter=0
            while(i<99):
                #print("counter: %s" %counter)
                #print("I: %s" %i)

                selected_reply_memory_previous_states[i] = reply_memory_previous_states[counter].reshape(1, 240)
                selected_reply_memory_previous_actions[i] = reply_memory_previous_actions[counter]
                selected_reply_memory_rewards[i] = reply_memory_rewards[counter]
                selected_reply_memory_current_states[i] = reply_memory_current_states[counter].reshape(1, 240)
                selected_reply_memory_q_values[i] = reply_memory_q_values[counter]

                counter = counter + 1
                if(counter>=reply_memory_counter):
                    counter=0

                i=i+1



        if(Reply_Memory_New_On_Update==True):
            for r in range(0,99):

                #randomly selecting among
                #i=0
                #while(i<31):
                #    random_number = np.random.randint(0, 320)  # we have three actions so we choose between them


                #updating values

                #Initialization

                previous_state = selected_reply_memory_previous_states[r].reshape(1, 240)
                previous_action=selected_reply_memory_previous_actions[r]
                reward=selected_reply_memory_rewards[r]
                current_state = selected_reply_memory_current_states[r].reshape(1, 240)

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
        reply_memory_inputs[99]=state
        reply_memory_outputs[99]=state_action_value_list


        #print("Reply Memory Outputs: %s" %reply_memory_outputs)

        model.fit(reply_memory_inputs[0:100, :].reshape(100,240),
                  reply_memory_outputs[0:100, :].reshape(100, 3), nb_epoch=100,
                  batch_size=100, verbose=0)

        reply_memory_index = reply_memory_index + 1

        if (reply_memory_index > 319):
            reply_memory_index = 0




    return model

def predict_3actions_DepthP1(model, state):

    return model.predict(state.reshape(1,240), batch_size=1)

################################# Kobuki movement functions

def kobuki_move_forward(howmany):
    cmd.linear.x = 0.40
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.0
    #print("Sensors:", sensors)


    safe=True

    if(safe):
        if(distanceCenter>40):
            if(distanceLeft>40 and distanceRight>40):
                velocity_publisher.publish(cmd)
                #rospy.sleep(0.5)
        if (distanceCenter > 40):
            if (distanceLeft > 40 and distanceRight > 40):
                velocity_publisher.publish(cmd)
                #rospy.sleep(0.5)
        if (distanceCenter > 40):
            if (distanceLeft > 40 and distanceRight > 40):
                velocity_publisher.publish(cmd)
                #rospy.sleep(ros_sleep_time)

    #print "ros_sleep_time: %s " %(ros_sleep_time,),

    else:
        velocity_publisher.publish(cmd)
        #rospy.sleep(0.5)
        velocity_publisher.publish(cmd)
        #rospy.sleep(0.5)
        velocity_publisher.publish(cmd)
        #rospy.sleep(ros_sleep_time)


    return True

def kobuki_turn_left(howmany):
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 1.0 #0.17

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_turn_right(howmany):
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -1.0
    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_turn_left_slow(howmany):
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.6 #0.17

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_turn_right_slow(howmany):
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -0.6 #0.17

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_turn_left_fast(howmany):
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 2.6  # 0.17
    #cmd.angular.z = 3.0

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_turn_right_fast(howmany):
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -2.6 #0.17
    #cmd.angular.z = -3.0

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_rotate_right_180(howmany):
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -5.0 #0.17

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_move_forward_slow(howmany):
    cmd.linear.x = 0.25
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.0

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

def kobuki_move_backward(howmany):
    cmd.linear.x = -0.5
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.0

    velocity_publisher.publish(cmd)
    #rospy.sleep(ros_sleep_time)
    return True

################################## reinfocement learning functions

def update_the_current_state_DepthP1():
    global current_state_DP1, previous_state_DP1, sensor_DP1
    global previous_state_weighted_reward, current_state_weighted_reward, weighted_reward

    previous_state_DP1=np.array(current_state_DP1)

    current_state_DP1=np.array(sensor_DP1)

    previous_state_weighted_reward=np.array(current_state_weighted_reward)

    current_state_weighted_reward=np.array(weighted_reward)

    return 0

def get_the_state():
    global previous_state, new_state, current_state, number_of_states, number_of_action_values, action, action_value, Alpha, Lambda

    state=current_state

    if (distanceLeft <= 150):
        sensorLeft2 =1
    else:
        sensorLeft2=0

    if (distanceLeft <= 95):
        sensorLeft1=1
    else:
        sensorLeft1=0

    if (distanceLeft <= 40):
        sensorLeft0=1
    else:
        sensorLeft0=0

    #check the center sensor
    if (distanceCenter <= 150):
        sensorCenter2 = 1
    else:
        sensorCenter2 = 0

    if (distanceCenter <= 95):
        sensorCenter1 = 1
    else:
        sensorCenter1 = 0

    if (distanceCenter <= 40):
        sensorCenter0 = 1
    else:
        sensorCenter0 = 0

    #check the right sensor
    if (distanceRight <= 150):
        sensorRight2 = 1
    else:
        sensorRight2 = 0

    if (distanceRight <= 95):
        sensorRight1 = 1
    else:
        sensorRight1 = 0

    if (distanceRight <= 40):
        sensorRight0 = 1
    else:
        sensorRight0 = 0


    #check the left 5 sensor
    if (distanceLeft5 <= 95):
        sensorLeft51 = 1
    else:
        sensorLeft51 = 0

    if (distanceLeft5 <= 40):
        sensorLeft50 = 1
    else:
        sensorLeft50 = 0


    #check the right 5 sensor
    if (distanceRight5 <= 95):
        sensorRight51 = 1
    else:
        sensorRight51 = 0

    if (distanceRight <= 40):
        sensorRight50 = 1
    else:
        sensorRight50 = 0

    state=(sensorLeft0,sensorLeft1,sensorLeft2,sensorCenter0,sensorCenter1,sensorCenter2,sensorRight0,sensorRight1,sensorRight2, sensorLeft50,sensorLeft51,sensorRight50,sensorRight51)

    return state

def update_the_current_state():
    global previous_state, new_state, current_state, number_of_states, number_of_action_values, action, action_value, Alpha, Lambda

    previous_state=current_state

    #check the left sensor
    if (distanceLeft <= 150):
        sensorLeft2 =1
    else:
        sensorLeft2=0

    if (distanceLeft <= 95):
        sensorLeft1=1
    else:
        sensorLeft1=0

    if (distanceLeft <= 40):
        sensorLeft0=1
    else:
        sensorLeft0=0

    #check the center sensor
    if (distanceCenter <= 150):
        sensorCenter2 = 1
    else:
        sensorCenter2 = 0

    if (distanceCenter <= 95):
        sensorCenter1 = 1
    else:
        sensorCenter1 = 0

    if (distanceCenter <= 40):
        sensorCenter0 = 1
    else:
        sensorCenter0 = 0

    #check the right sensor
    if (distanceRight <= 150):
        sensorRight2 = 1
    else:
        sensorRight2 = 0

    if (distanceRight <= 95):
        sensorRight1 = 1
    else:
        sensorRight1 = 0

    if (distanceRight <= 40):
        sensorRight0 = 1
    else:
        sensorRight0 = 0

    current_state=(sensorLeft0,sensorLeft1,sensorLeft2,sensorCenter0,sensorCenter1,sensorCenter2,sensorRight0,sensorRight1,sensorRight2)

    return 0

def move_turtle_bot(turtlebot_action):
    if (turtlebot_action== 0):
        kobuki_move_forward(1)

    if(turtlebot_action== 1):
        kobuki_turn_left_slow(1)

    if (turtlebot_action == 2):
        kobuki_turn_right_slow(1)

    if(turtlebot_action == 3):
        kobuki_turn_left_fast(1)

    if (turtlebot_action == 4):
        kobuki_turn_right_fast(1)

    return 0

############################## Normal Q-Learning and SARSA - Artificial Neural Network - Using Depth Image - Phase 1
def get_reward_ANN_DepthP1(action, previous_state, current_state):
    accumulative_reward=0


    if(current_state[0]==1):
        accumulative_reward = accumulative_reward-200

    if(current_state[3]==1):
        accumulative_reward = accumulative_reward-150

    if(current_state[6]==1):
        accumulative_reward = accumulative_reward-150

    if(action==0):
        accumulative_reward = accumulative_reward + 50

    if (action==1 or action==2 or action == 3 or action == 4):
        accumulative_reward = accumulative_reward - 25

    '''

    # negative reward when the robot do something that still stays in a bad area
    if (previous_state[0] == 0 and current_state[0]==1): #left censor
        accumulative_reward = accumulative_reward-100

    if (previous_state[3] == 0 and current_state[3] == 1): #center censor
        accumulative_reward = accumulative_reward-100

    if (previous_state[6] == 0 and current_state[6]==1): #right sensor
        accumulative_reward = accumulative_reward-100

    #if (previous_state[9] == 0 and current_state[9]==1): #left5 censor
    #    accumulative_reward = accumulative_reward-0

    #if (previous_state[11] == 0 and current_state[11]==1): #right5 censor
    #    accumulative_reward = accumulative_reward-0



    if (previous_state[0] == 1 and current_state[0]==1): #left censor
        accumulative_reward=accumulative_reward-100

    if (previous_state[3] == 1 and current_state[3]==1): #center censor
        accumulative_reward=accumulative_reward-100

    if (previous_state[6] == 1 and current_state[6]==1): #left censor
        accumulative_reward=accumulative_reward-100

    #if (previous_state[9] == 1 and current_state[9]==1): #left censor
    #    accumulative_reward=accumulative_reward-0

    #if (previous_state[11] == 1 and current_state[11]==1): #left censor
    #    accumulative_reward=accumulative_reward-0


    #positive reward when robot does something that realeases from a bad area
    if (previous_state[0] == 1 and current_state[0] == 0):  # left censor
        accumulative_reward = accumulative_reward + 100

    if (previous_state[3] == 1 and current_state[3] == 0):  # center censor
        accumulative_reward = accumulative_reward + 100

    if (previous_state[6] == 1 and current_state[6] == 0):  # right sensor
        accumulative_reward = accumulative_reward + 100

    #if (previous_state[9] == 1 and current_state[9] == 0):  # left5 sensor
    #    accumulative_reward = accumulative_reward + 45

    #if (previous_state[11] == 1 and current_state[11] == 0):  # right5 sensor
    #    accumulative_reward = accumulative_reward + 45

    ##################################################################### second level sensor

    if (previous_state[1] == 0 and current_state[1]==1): #left censor
        accumulative_reward = accumulative_reward-50

    if (previous_state[4] == 0 and current_state[4] == 1): #center censor
        accumulative_reward = accumulative_reward-50

    if (previous_state[7] == 0 and current_state[7]==1): #right sensor
        accumulative_reward = accumulative_reward-50

    #if (previous_state[10] == 0 and current_state[10] == 1):  # left5 sensor
    #    accumulative_reward = accumulative_reward + 0

    #if (previous_state[12] == 0 and current_state[12] == 1):  # right5 sensor
    #    accumulative_reward = accumulative_reward + 0


    if (previous_state[1] == 1 and current_state[1]==1): #left censor
        accumulative_reward=accumulative_reward-50

    if (previous_state[4] == 1 and current_state[4]==1): #center censor
        accumulative_reward=accumulative_reward-50

    if (previous_state[7] == 1 and current_state[7]==1): #left censor
        accumulative_reward=accumulative_reward-50


    #positive reward when robot does something that realeases from a bad area
    if (previous_state[1] == 1 and current_state[1] == 0):  # left censor
        accumulative_reward = accumulative_reward + 50

    if (previous_state[4] == 1 and current_state[4] == 0):  # center censor
        accumulative_reward = accumulative_reward + 50

    if (previous_state[7] == 1 and current_state[7] == 0):  # right sensor
        accumulative_reward = accumulative_reward + 50

    #if (previous_state[10] == 1 and current_state[10] == 0):  # left5 sensor
    #    accumulative_reward = accumulative_reward + 25

    #if (previous_state[12] == 1 and current_state[12] == 0):  # right5 sensor
    #    accumulative_reward = accumulative_reward + 25

    ##################################################################### third level sensor

    if (previous_state[2] == 0 and current_state[2] == 1):  # left censor
        accumulative_reward = accumulative_reward - 25

    if (previous_state[5] == 0 and current_state[5] == 1):  # center censor
        accumulative_reward = accumulative_reward - 25

    if (previous_state[8] == 0 and current_state[8] == 1):  # right sensor
        accumulative_reward = accumulative_reward - 25



    if (previous_state[2] == 1 and current_state[2] == 1):  # left censor
        accumulative_reward = accumulative_reward - 25

    if (previous_state[5] == 1 and current_state[5] == 1):  # center censor
        accumulative_reward = accumulative_reward - 25

    if (previous_state[8] == 1 and current_state[8] == 1):  # left censor
        accumulative_reward = accumulative_reward - 25



    # positive reward when robot does something that realeases from a bad area
    if (previous_state[2] == 1 and current_state[2] == 0):  # left censor
        accumulative_reward = accumulative_reward + 25

    if (previous_state[5] == 1 and current_state[5] == 0):  # center censor
        accumulative_reward = accumulative_reward + 25

    if (previous_state[8] == 1 and current_state[8] == 0):  # right sensor
        accumulative_reward = accumulative_reward + 25

            ###################################################################### reward for going forward

    #robot get rewards if it goes foward when there is no near obstacle
    if(action==0):
        if (previous_state[0] == 0 and previous_state[3] == 0 and previous_state[6] == 0):
            if (previous_state[1] == 0 and previous_state[4] == 0 and previous_state[7] == 0):
                if (previous_state[2] == 0 and previous_state[5] == 0 and previous_state[8] == 0):
                    accumulative_reward = accumulative_reward + 100
                #else:
                    #accumulative_reward = accumulative_reward + 0.2
            #else:
                #accumulative_reward = accumulative_reward + 0.1
    elif(action==1 or action==2 or action == 3 or action == 4):
        if (previous_state[0] == 0 and previous_state[3] == 0 and previous_state[6] == 0):
            if (previous_state[1] == 0 and previous_state[4] == 0 and previous_state[7] == 0):
                if (previous_state[2] == 0 and previous_state[5] == 0 and previous_state[8] == 0):
                    accumulative_reward = accumulative_reward - 100 #from 55
                else:
                    accumulative_reward = accumulative_reward - 40 #from 40
            #else:
                #accumulative_reward = accumulative_reward - 0.2

    #else:
    #    if (action == 3 or action == 4):
    #        if (previous_state[0] == 0 and previous_state[3] == 0 and previous_state[6] == 0):
    #            accumulative_reward = accumulative_reward - 100



            #print("Previous State: %s" % (previous_state,))
    #print("Action Selected: %s" % (action,))
    #print("Current State: %s" % (current_state,))
    #print("Accumulative reward: %s" % (accumulative_reward,))

    #weighted_previous_state=np.sum(previous_state_weighted_reward)
    #weighted_current_state = np.sum(current_state_weighted_reward)
    #weighted_reward=-((weighted_current_state)-(weighted_previous_state-weighted_current_state))


    #if(action==0):
       # weighted_reward=weighted_reward+800


    #print("Weighted Value Previous State: ", weighted_previous_state)
    #print("Weighted Value Current State: ", weighted_current_state)
    #print("Weighted Reward: ", weighted_reward)

    '''

    '''
    ### weighted reward for three sensors
    accumulative_reward=0

    if (current_state[1] == 1 or current_state[1] == 1 or current_state[1] == 1):
        centerReward=(distanceCenter-40) * 3
        leftReward=(distanceLeft-40) * 2
        rightReward=(distanceRight-40) * 2

        accumulative_reward=centerReward + leftReward + rightReward
    else:
        if(action==0):
            accumulative_reward = accumulative_reward + 20
        else:
            accumulative_reward = accumulative_reward - 20
    '''

    return accumulative_reward

def make_move_ANN_DepthP1(current_state,model,epsilon):
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

    #action=0

    move_turtle_bot(action)


    return action

def update_Q_value_ANN_DepthP1(previous_state,previous_action, current_state,reward,model,epsilon):
    global Z_S_A, Z_S_A_index, Z_S_A_counter, N_Step_On,Alpha,Lambda, Gamma, Z_S_A_inputs, Z_S_A_outputs,QLearning


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
                               epsilon)

    else:
        print "4Steps, TD(Lambda)",

    ############################################################

    previous_possible_actions = predict_3actions_DepthP1(model, previous_state)

    print "After: %.2f,%.2f,%.2f," % (previous_possible_actions[0][0],previous_possible_actions[0][1],previous_possible_actions[0][2]),#,previous_possible_actions[3],previous_possible_actions[4],),


############################################ start

def start():
    global distanceLeft, distanceCenter, distanceRight,distanceLeft5, distanceRight5, turtlebot_state_action_value, previous_state,current_state, bridge, current_state_DP1, previous_state_DP1, sensor_DP1, current_state_weighted_reward, previous_state_weighted_reward, image_DP1
    global Reply_Memory_On, Reply_Memory_New_On, N_Step_On, Alpha,Lambda, Gamma, Reply_Memory_New_On_Update
    global ros_sleep_time, QLearning

    #######################################################3 variables

    previous_state_DP1_temp = np.zeros((1, 240), np.float32)
    previous_state_DP1 = np.zeros((1, 240), np.float32)

    previous_state_3ultrasonic = np.zeros((1, 3), np.int32)
    previous_state_3ultrasonic_all = np.zeros((20001, 3), np.int32)
    previous_state_3ultrasonic_5states = np.zeros((1, 15), np.int32)
    previous_state_3ultrasonic_5states_all = np.zeros((20001, 15), np.int32)

    previous_depth_image=np.zeros((1, 80, 60), np.float32)
    previous_depth_image_all = np.zeros((20001, 1, 80, 60), np.float32)
    previous_depth_image_5states = np.zeros((5, 80, 60), np.float32)
    previous_depth_image_5states_all = np.zeros((20001, 5, 80, 60), np.float32)

    action_all = np.zeros((20001, 1), np.int32)
    reward_all = np.zeros((20001, 1), np.float32)

    current_state_DP1_temp = np.zeros((1, 240), np.float32)
    current_state_DP1 = np.zeros((1, 240), np.float32)

    current_state_3ultrasonic = np.zeros((1, 3), np.int32)
    current_state_3ultrasonic_all = np.zeros((20001, 3), np.int32)
    current_state_3ultrasonic_5states = np.zeros((1, 15), np.int32)
    current_state_3ultrasonic_5states_all = np.zeros((20001, 15), np.int32)

    current_depth_image = np.zeros((1, 80, 60), np.float32)
    current_depth_image_all=np.zeros((20001, 1, 80, 60), np.float32)

    current_depth_image_5states = np.zeros((5, 80, 60), np.float32)
    current_depth_image_5states_all = np.zeros((20001, 5, 80, 60), np.float32)


    ######################################################## call backs

    distanceLeft=0
    distanceCenter=0
    distanceRight=0
    distanceLeft5=0
    distanceRight5=0
    rospy.Subscriber("sonar_sensorL1", LaserScan, chatterCallback_sonarSensorL1)
    rospy.Subscriber("sonar_sensor", LaserScan, chatterCallback_sonarSensor)
    rospy.Subscriber("sonar_sensorR1", LaserScan, chatterCallback_sonarSensorR1)
    #rospy.Subscriber("sonar_sensorL5", LaserScan, chatterCallback_sonarSensorL5)
    #rospy.Subscriber("sonar_sensorR5", LaserScan, chatterCallback_sonarSensorR5)

    bridge = CvBridge()

    rospy.Subscriber("/camera/depth/image_raw", Image, image_callback) # for kinect

    #rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback_RGB)  # for kinect

    #######################################################################

    ################ for saving multi dimensional arrays


    outfile = TemporaryFile()

    # x = np.arange(10)
    #outfile.name="wwwe.txt"
    #np.save(outfile, previous_depth_image_all)
    np.save("wwwe.txt", previous_depth_image_all)

    ################



    print("Starting application...")
    print("Sensors Value: %s" %(sensors,))
    print("Number of states: %s" %(number_of_states))
    print("Number of action values: %s" %(number_of_action_values))

    create_weighting_matrix()
    #create_weighting_matrix_240()
    print("Weighting Matrix Created.")

    #rospy.sleep(ros_sleep_time * 2) #waiting for the sonar sensors

    print("initialization...")

    update_the_current_state()

    load_model_scenario=False
    Load_And_Train_Using_Old_Model=False

    max_counter = 20000.0
    max_counter_test = 1000.0
    Reply_Memory_On = True
    Reply_Memory_New_On = False
    Reply_Memory_New_On_Update = False
    N_Step_On = False
    QLearning=True
    NumberOfActions=3
    ReplyMemoryCapacity=100
    MemoryCapacity=5
    Keras_Optimizer='rms'
    Keras_Learning_Rate=0.002

    #filename_str = "model-2000-5-QLearning-MemoryBasedPolicy-Adamax-3ultra-Lambda-10000"
    filename_str = 'MMRL-V0.5-2-' + str(QLearning) + '-' + str(Alpha) + '-' + str(Gamma) + '-' + str(Lambda) + '-FA-' + str(Keras_Optimizer) + '-' + str(Keras_Learning_Rate) +'-MaxCounter-' + str(max_counter) + '-S-RMO-' + str(Reply_Memory_On) + '-RMNO-' + str(Reply_Memory_New_On) + '-MSO-' + str(N_Step_On) + '-A-' + str(NumberOfActions) + '-RM-' + str(ReplyMemoryCapacity) + '-M-' + str(MemoryCapacity)

    if(load_model_scenario==True):

        print("loading model...")

        # load json and create model
        json_file = open(filename_str + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(filename_str + ".h5")

        print("Loaded model from disk")

        i = 0
        reward=0
        running_reward=0
        plt.xlabel("Steps")
        plt.ylabel("Avg Reward")
        plt.axis([0, max_counter,-350, 100])
        ax = plt.gca()

        major_ticks_x = np.arange(0, max_counter, max_counter/10)
        major_ticks_y = np.arange(-350, 100, 10)

        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)

        ax.grid(which='both')

        epsilon=0.0

        while not i >= max_counter_test:

            print("Step: %s," %(i,)),

            #previous_state_3ultrasonic = np.array((distanceLeft, distanceRight, distanceCenter)).reshape(1,3)
            #previous_state_3ultrasonic = np.array((float(distanceLeft)/1, float(distanceRight)/1, float(distanceCenter)/1)).reshape(1, 3)
            previous_state_3ultrasonic = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            previous_state_3ultrasonic_5states[0, 3:15]=np.array(previous_state_3ultrasonic_5states[0, 0:12])
            previous_state_3ultrasonic_5states[0, 0:3] = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            #previous_state_DP1 = np.array(sensor_DP1)

            previous_state_DP1[0, 80:240]=np.array(previous_state_DP1[0,0:160])
            previous_state_DP1[0, 0:80]=np.array(sensor_DP1)

            previous_depth_image=image_DP1

            previous_depth_image_5states[1:5]=previous_depth_image_5states[0:4]
            previous_depth_image_5states[0]=image_DP1.reshape(1,80,60)

            previous_state = get_the_state()

            previous_state_weighted_reward = np.array(weighted_reward)

            # print("Main Loop")
            # print("Previous State: ", previous_state_DP1)
            # print("Previous State DP1: ", previous_state)
            # print("Previous State Weighted Reward: ", previous_state_weighted_reward)
            print("State Ultrasonic: %s" %(previous_state_3ultrasonic_5states[0, 0:3],)),

            # for checking
            action = make_move_ANN_DepthP1(previous_state_DP1, model, epsilon)
            #action = make_move_ANN_DepthP1(previous_state_3ultrasonic, model, epsilon)
            #action = make_move_ANN_DepthP1(previous_state_DP1, model, epsilon)

            rospy.sleep(ros_sleep_time)


            print(",Action: %s" %(action,))


            current_state_3ultrasonic_5states[0, 3:15]=np.array(current_state_3ultrasonic_5states[0, 0:12])
            current_state_3ultrasonic_5states[0, 0:3] = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            current_state_DP1[0, 80:240]=np.array(current_state_DP1[0,0:160])
            current_state_DP1[0, 0:80]=np.array(sensor_DP1)
            #current_state_DP1 = np.array(sensor_DP1)

            current_depth_image = image_DP1

            current_depth_image_5states[1:5]=current_depth_image_5states[0:4]
            current_depth_image_5states[0]=image_DP1.reshape(1,80,60)


            current_state = get_the_state()

            current_state_weighted_reward = np.array(weighted_reward)

            # print("Main Loop")
            # print("Current State: ",current_state)
            # print("Current State DP1: ", current_state_DP1)
            # print("Current State Weighted Reward: ", current_state_weighted_reward)
            #print("Current State Ultrasonice: ", current_state_3ultrasonic)

            reward = get_reward_ANN_DepthP1(action, previous_state, current_state)

            ###################

            running_reward=running_reward+reward

            i = i + 1

            running_reward_mean=running_reward/i

            plt.plot(i, running_reward_mean, marker='.', color='b')

            plt.pause(0.05)

        print("testing finished.")

    else:

        if(Load_And_Train_Using_Old_Model==False):
            print("creating model...")
            print("K image dim ordering: %s" %(K.image_dim_ordering(),))

            img_rows=60
            img_cols=80

            if K.image_dim_ordering() == 'th':
                input_shape = (5, img_rows, img_cols)
            else:
                input_shape = (img_rows, img_cols, 5)

            model = Sequential()

            model.add(Dense(1000, input_dim=240, init='normal', activation='linear'))
            model.add(Dropout(0.5))

            #model.add(Dense(500, init='normal', activation='relu'))
            #model.add(Dropout(0.5))

            model.add(Dense(1000, init='normal', activation='relu'))
            model.add(Dropout(0.5))

            model.add(Dense(3, init='normal', activation='linear'))
            model.add(Dropout(0.5))

            #print(model.summary())
            #print(model.optimizer[0])

            adamax = Adamax(lr=Keras_Learning_Rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # learning rate (default) = 0.002

            rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

            #model.compile(loss='mse', optimizer=adamax)
            model.compile(loss='mse', optimizer=adamax)

            print("model created...")
        else:
            print("load and train the model...")

            # load json and create model
            json_file = open(filename_str + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(filename_str + ".h5")

            adamax = Adamax(lr=Keras_Learning_Rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # learning rate (default) = 0.002

            rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

            model.compile(loss='mse', optimizer=rms)

            print("Loaded model from disk")

        epsilon=0.1
        i=0
        running_reward=0
        plt.xlabel("Steps")
        plt.ylabel("Avg Reward")
        plt.axis([0, max_counter,-350, 100])

        ax = plt.gca()

        major_ticks_x = np.arange(0, max_counter, max_counter/10)
        major_ticks_y = np.arange(-350, 100, 10)

        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)

        ax.grid(which='both')

        epsilon=0.1
        i=0
        while not i>=max_counter:

            print("")
            #print("")
            print("%s:" %(i,)),


            if(i<(max_counter/10-(max_counter/100))):
                epsilon=1-(i/(max_counter/10))
            #else:
                #epsilon=0.9
            else:
                epsilon=0.1

            #print(i/(max_counter/10))

            #previous_state_3ultrasonic = np.array((distanceLeft, distanceRight, distanceCenter)).reshape(1,3)
            previous_state_3ultrasonic = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            previous_state_3ultrasonic_5states[0, 3:15]=np.array(previous_state_3ultrasonic_5states[0, 0:12])
            previous_state_3ultrasonic_5states[0, 0:3] = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            previous_state_DP1[0, 80:240]=np.array(previous_state_DP1[0,0:160])
            previous_state_DP1[0, 0:80]=np.array(sensor_DP1)

            previous_depth_image=image_DP1

            previous_depth_image_5states[1:5]=previous_depth_image_5states[0:4]
            previous_depth_image_5states[0]=image_DP1.reshape(1,80,60)

            #print(sensor_DP1)
            #print("previous: ")
            #print(image_DP1)
            #print("Input sample shape: " )
            #print(previous_depth_image.shape)

            #previous_state_DP1 = np.array(sensor_DP1)

            previous_state = get_the_state()

            previous_state_weighted_reward = np.array(weighted_reward)

            # print("Main Loop")
            # print("Previous State: ", previous_state_DP1)
            # print("Previous State DP1: ", previous_state)
            # print("Previous State Weighted Reward: ", previous_state_weighted_reward)
            #print("Previous State Ultrasonice: ", previous_state_3ultrasonic)

            # for checking
            #action = make_move_ANN_DepthP1(previous_state_DP1, model, epsilon)
            #action = make_move_ANN_DepthP1(previous_state_3ultrasonic, model, epsilon)

            #print("Before predicting...")

            #action = make_move_ANN_DepthP1(previous_depth_image, model, epsilon)
            action = make_move_ANN_DepthP1(previous_state_DP1, model, epsilon)

            #print("Before training...")


            rospy.sleep(ros_sleep_time)


            #action=0
            #print("Action: ", action)

            current_state_3ultrasonic = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            current_state_3ultrasonic_5states[0, 3:15]=np.array(current_state_3ultrasonic_5states[0, 0:12])
            current_state_3ultrasonic_5states[0, 0:3] = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            current_state_DP1[0, 80:240]=np.array(current_state_DP1[0,0:160])
            current_state_DP1[0, 0:80]=np.array(sensor_DP1)

            current_depth_image = image_DP1

            current_depth_image_5states[1:5]=current_depth_image_5states[0:4]
            current_depth_image_5states[0]=image_DP1.reshape(1,80,60)

            #print("test print: ")
            #print(current_depth_image_5states)

            #print("current: ")
            #print(image_DP1)
            #current_state_DP1 = np.array(sensor_DP1)

            current_state = get_the_state()

            current_state_weighted_reward = np.array(weighted_reward)

            # print("Main Loop")
            # print("Current State: ",current_state)
            # print("Current State DP1: ", current_state_DP1)
            # print("Current State Weighted Reward: ", current_state_weighted_reward)
            #print("Current State Ultrasonice: ", current_state_3ultrasonic)

            reward = get_reward_ANN_DepthP1(action, previous_state, current_state)

            #print("Action : %s, Reward: %s, Pre-Ul: %s, Cur-Ul: %s, " % (action, reward,previous_state_3ultrasonic,current_state_3ultrasonic,)),
            print("Action: %s, Pre-Ul: %i, %i, %i, Cur-Ul: %i, %i, %i, Reward:%s," % (action,previous_state_3ultrasonic[0][0], previous_state_3ultrasonic[0][1], previous_state_3ultrasonic[0][2],current_state_3ultrasonic[0][0], current_state_3ultrasonic[0][1], current_state_3ultrasonic[0][2], reward, )),


            running_reward=running_reward+reward

            #update_Q_value_ANN_DepthP1(previous_state_DP1, action, current_state_DP1, reward,model,epsilon)
            #update_Q_value_ANN_DepthP1(previous_state_3ultrasonic, action, current_state_3ultrasonic, reward, model, epsilon)

            #update_Q_value_ANN_DepthP1(previous_depth_image, action, current_depth_image, reward, model,
            #                           epsilon)

            update_Q_value_ANN_DepthP1(previous_state_DP1, action, current_state_DP1, reward, model,
                                       epsilon)

            #previous_depth_image_all[i] = previous_depth_image.reshape(1, 80, 60)
            #previous_depth_image_5states_all[i] = previous_depth_image_5states.reshape(5, 80, 60)
            #previous_state_3ultrasonic_all[i] = previous_state_3ultrasonic
            #previous_state_3ultrasonic_5states_all[i]=previous_state_3ultrasonic_5states

            #action_all[i] = action
            #reward_all[i] = reward

            #current_depth_image_all[i] = current_depth_image.reshape(1,80,60)
            #current_depth_image_5states_all[i] = current_depth_image_5states.reshape(5, 80, 60)
            #current_state_3ultrasonic_all[i] = current_state_3ultrasonic
            #current_state_3ultrasonic_5states_all[i] = current_state_3ultrasonic_5states

            #print("")
            print("")

            i = i + 1

            #runningMean = np.mean(av[:, 1])
            running_reward_mean=running_reward/i

            #plt.scatter(i, running_reward_mean)
            #plt.plot(i, running_reward_mean,linestyle = 'dashed',linewidth=2,color='r')

            #plt.plot(i, running_reward_mean,marker = '.',markersize=2,linestyle = '-',linewidth=2,color='b')
            plt.plot(i, running_reward_mean, marker='.', color='b')
            #, linewidth=0.2)
            plt.pause(0.05)

        #plt.show()
        '''
        np.savetxt(filename_str + '-data-action.out', action_all)
        np.savetxt(filename_str + '-data-reward.out', reward_all)


        ################ for saving multi dimensional arrays

        np.save(filename_str + '-data-previous_depth_image', previous_depth_image_all)
        np.save(filename_str + '-data-current_depth_image', current_depth_image_all)
        np.save(filename_str + '-data-previous_image_5states', previous_depth_image_5states_all)
        np.save(filename_str + '-data-current_image_5states', current_depth_image_5states_all)

        #for loadiong this can be useful
        #np.load(outfile)

        ################



        np.savetxt(filename_str + '-data-previous_state_3ultrasonic.out', previous_state_3ultrasonic_all)
        np.savetxt(filename_str + '-data-current_state_3ultrasonic.out', current_state_3ultrasonic_all)
        np.savetxt(filename_str + '-data-previous_state_3ultrasonic_5states.out', previous_state_3ultrasonic_5states_all)
        np.savetxt(filename_str + '-data-current_state_3ultrasonic_5states.out', current_state_3ultrasonic_5states_all)
        '''



        plt.savefig(filename_str + '.jpg')

        print("training finished.")

        # serialize model to JSON
        model_json = model.to_json()
        with open(filename_str + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(filename_str + ".h5")
        print("Saved model to disk")



if __name__ == '__main__':
    try:
	init()
        start()
    except rospy.ROSInterruptException:
        pass
