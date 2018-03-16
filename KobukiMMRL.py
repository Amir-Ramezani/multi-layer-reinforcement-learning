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

## more to opencv

import time

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tempfile import TemporaryFile

from NeuralNetwork import create_model, load_model
from ReinforcementLearning import get_reward_ANN_DepthP1, make_move_ANN_DepthP1, update_Q_value_ANN_DepthP1

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

#sensor_DP1 = np.zeros((1, 160), np.float32)
#sensor_DP1 = np.zeros((1, 240), np.float32)

global weight_matrix
weight_matrix = np.zeros((256, 80, 1), np.double)
#weight_matrix = np.zeros((256, 160, 1), np.double)
#weight_matrix = np.zeros((256, 240, 1), np.float32)


global weighted_reward, previous_state_weighted_reward, current_state_weighted_reward
weighted_reward = np.zeros((1, 80), np.int32)
#weighted_reward = np.zeros((1, 160), np.int32)
#weighted_reward = np.zeros((1, 240), np.float32)
previous_state_weighted_reward = np.zeros((1, 80), np.int32)
#previous_state_weighted_reward = np.zeros((1, 160), np.int32)
#previous_state_weighted_reward = np.zeros((1, 240), np.float32)
current_state_weighted_reward = np.zeros((1, 80), np.int32)
#current_state_weighted_reward = np.zeros((1, 160), np.int32)
#current_state_weighted_reward = np.zeros((1, 240), np.float32)

obstacle=True
obstacle_avoided=True
publish=False

global turtlebot_state

turtlebot_state=[]

global ros_sleep_time


################################# Initialization

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

def init_turtlebot_state():
    global previous_state,new_state,current_state,number_of_states,number_of_action_values,action,action_value,Alpha,Lambda, Gamma, previous_state_DP1, current_state_DP1

    previous_state_DP1 = np.zeros((1, 640), np.uint8) #memory based policy

    #previous_state_DP1 = np.zeros((1, 80), np.float32)
    #previous_state_DP1 = np.zeros((1, 160), np.float32)
    #previous_state_DP1 = np.zeros((1, 240), np.float32)

    current_state_DP1 = np.zeros((1, 640), np.uint8) #memory based policy

    #current_state_DP1 = np.zeros((1, 80), np.float32)
    #current_state_DP1 = np.zeros((1, 160), np.float32)
    #current_state_DP1 = np.zeros((1, 240), np.float32)

    #previous_state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
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
        #turtlebot_state_action_value[(state, 3)] = 0.0
        #turtlebot_state_action_value[(state, 4)] = 0.0
        #turtlebot_state_action_value[(state, 5)] = 0.0 #turn left fast
        #turtlebot_state_action_value[(state, 6)] = 0.0 #turn right fast


        number_of_action_values=number_of_action_values+3



    return 0

################################# Other functions

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

################################# Callback functions

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
    #sensors[0]=distanceLeft
    #sensors[1]=distanceCenter
    #sensors[2]=distanceRight
    #sensors[3] = distanceLeft5
    #sensors[4] = distanceRight5
    #print(sensors)

def chatterCallback_sonarSensorL5(msg):
    global distanceLeft, distanceCenter, distanceRight, distanceLeft5, distanceRight5
    distanceLeft5=int((msg.ranges[0])*100)
    #sensors[0]=distanceLeft
    #sensors[1]=distanceCenter
    #sensors[2]=distanceRight
    #sensors[3] = distanceLeft5
    #sensors[4] = distanceRight5

def chatterCallback_sonarSensorR5(msg):
    global distanceLeft, distanceCenter, distanceRight, distanceLeft5, distanceRight5
    distanceRight5 = int((msg.ranges[0]) * 100)
    #sensors[0] = distanceLeft
    #sensors[1] = distanceCenter
    #sensors[2] = distanceRight
    #sensors[3] = distanceLeft5
    #sensors[4] = distanceRight5
    #print(sensors)

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
            if(distanceLeft < 50):
                cv_image_resized[:,i] = (float)(distanceLeft + 5)/100
            else:
                #cv_image_resized[np.nan,5]=0.001
                cv_image_resized[np.isnan(cv_image_resized[:,i]),i]=5.0
                #min = np.min(cv_image_resized[:, i])
                #print("min is: ")
                #print(min)
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

    '''
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
            #print(min)
            blank_image_resized[min, i] = 255
            #sensor_DP1[0, i] = min
            #tmp=
            #tmp.astype(np.int)
            #print(sensor_DP1)
            #time.sleep(1)
            val=(int(weight_matrix[min,i])*255)/100
            weighted_reward[0,i]=val
        except:
            print(min)
    '''

    sensor_DP1 = np.array(image_DP1[25, :] * 50)

    #cv2.imshow("Blank Image Resized", blank_image_resized)
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

#####################################



#####################################

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

############################################ start

def start():
    global distanceLeft, distanceCenter, distanceRight,distanceLeft5, distanceRight5, turtlebot_state_action_value, previous_state,current_state, bridge, current_state_DP1, previous_state_DP1, sensor_DP1, current_state_weighted_reward, previous_state_weighted_reward, image_DP1
    global Reply_Memory_On, Reply_Memory_New_On, Alpha,Lambda, Gamma, Reply_Memory_New_On_Update
    global ros_sleep_time, QLearning

    #######################################################3 variables

    previous_state_DP1_temp = np.zeros((1, 640), np.float32)
    previous_state_DP1 = np.zeros((1, 640), np.float32)

    previous_state_3ultrasonic = np.zeros((1, 3), np.float32)
    #previous_state_3ultrasonic_all = np.zeros((100001, 3), np.int32)
    previous_state_3ultrasonic_5states = np.zeros((1, 15), np.float32)
    #previous_state_3ultrasonic_5states_all = np.zeros((100001, 15), np.int32)

    previous_depth_image=np.zeros((1, 80, 60), np.float32)
    #previous_depth_image_all = np.zeros((100001, 1, 80, 60), np.float32)
    previous_depth_image_5states = np.zeros((4, 80, 60), np.float32)
    #previous_depth_image_5states_all = np.zeros((100001, 4, 80, 60), np.float32)

    action_all = np.zeros((50001, 1), np.uint8)
    reward_all = np.zeros((50001, 1), np.int16)

    q_values_all = np.zeros((50001, 3), np.float32)
    q_value = np.zeros((50001, 1), np.float32)

    current_state_DP1_temp = np.zeros((1, 640), np.float32)
    current_state_DP1 = np.zeros((1, 640), np.float32)

    current_state_3ultrasonic = np.zeros((1, 3), np.float32)
    #current_state_3ultrasonic_all = np.zeros((100001, 3), np.int32)
    current_state_3ultrasonic_5states = np.zeros((1, 15), np.float32)
    #current_state_3ultrasonic_5states_all = np.zeros((100001, 15), np.int32)

    current_depth_image = np.zeros((1, 80, 60), np.float32)
    #current_depth_image_all=np.zeros((100001, 1, 80, 60), np.float32)

    current_depth_image_5states = np.zeros((4, 80, 60), np.float32)
    #current_depth_image_5states_all = np.zeros((100001, 4, 80, 60), np.float32)


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
    #np.save("wwwe.txt", previous_depth_image_all)

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

    #update_the_current_state()

    load_model_scenario=True
    Load_And_Train_Using_Old_Model=False

    max_counter = 50000.0
    max_counter_test = 1000.0
    Reply_Memory_On = True
    Reply_Memory_New_On = True
    Reply_Memory_New_On_Update = True
    N_Step_On = False
    QLearning=True
    NumberOfActions=3
    ReplyMemoryCapacity=100
    MemoryCapacity=4
    #Keras_Optimizer='rms'
    #Keras_Learning_Rate=0.002

    EveryHowmanyFrames=4
    EveryHowmanyFrames_Counter=0

    #filename_str = "model-2000-5-QLearning-MemoryBasedPolicy-Adamax-3ultra-Lambda-10000"
    filename_str = 'V0.36-2-FixTheReward-ChangedModelRelu-MMRL-Batch-2-' + str(QLearning) + '-' + str(Alpha) + '-' + str(Gamma) + '-' + str(Lambda) + '-FA-' + str('rms') + '-' + str(0.002) +'-MaxCounter-' + str(max_counter) + '-S-RMO-' + str(Reply_Memory_On) + '-RMNO-' + str(Reply_Memory_New_On) + '-MSO-' + str(N_Step_On) + '-A-' + str(NumberOfActions) + '-RM-' + str(ReplyMemoryCapacity) + '-M-' + str(MemoryCapacity)

    if(load_model_scenario==True):

        print("loading model...")

        model=load_model(filename_str)

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

            previous_state_DP1[0, 80:640]=np.array(previous_state_DP1[0,0:560])
            previous_state_DP1[0, 0:80]=np.array(sensor_DP1)

            previous_depth_image=image_DP1

            previous_depth_image_5states[1:4]=previous_depth_image_5states[0:3]
            previous_depth_image_5states[0]=image_DP1.reshape(1,80,60)

            previous_state = get_the_state()

            previous_state_weighted_reward = np.array(weighted_reward)

            # print("Main Loop")
            # print("Previous State: ", previous_state_DP1)
            # print("Previous State DP1: ", previous_state)
            # print("Previous State Weighted Reward: ", previous_state_weighted_reward)
            print("State Ultrasonic: %s" %(previous_state_3ultrasonic_5states[0, 0:3],)),

            # for checking
            action = make_move_ANN_DepthP1(previous_state_DP1, model, epsilon,distanceCenter,distanceLeft,distanceRight)
            #action = make_move_ANN_DepthP1(previous_state_3ultrasonic, model, epsilon)
            #action = make_move_ANN_DepthP1(previous_state_DP1, model, epsilon)
            #ros_sleep_time=0.1
            rospy.sleep(ros_sleep_time)


            print(",Action: %s" %(action,))


            current_state_3ultrasonic_5states[0, 3:15]=np.array(current_state_3ultrasonic_5states[0, 0:12])
            current_state_3ultrasonic_5states[0, 0:3] = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            current_state_DP1[0, 80:640]=np.array(current_state_DP1[0,0:560])
            current_state_DP1[0, 0:80]=np.array(sensor_DP1)
            #current_state_DP1 = np.array(sensor_DP1)

            current_depth_image = image_DP1

            current_depth_image_5states[1:4]=current_depth_image_5states[0:3]
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

            #plt.pause(0.05)

        print("testing finished.")

    else:

        if(Load_And_Train_Using_Old_Model==False):
            #print("creating model...")

            model = create_model()

            #print("model created...")
        else:
            print("load and train the model...")

            model = load_model(filename_str)

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

            start_time_whole = time.time()

            print("")
            #print("")
            print("%s:" %(i,)),


            if(i<(max_counter/10-(max_counter/100))):
                epsilon=1-(i/(max_counter/10))
            #else:
                #epsilon=0.9
                #epsilon=0.05
            else:
                epsilon=0.1


            #previous_state_3ultrasonic = np.array((distanceLeft, distanceRight, distanceCenter)).reshape(1,3)
            previous_state_3ultrasonic = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            previous_state_3ultrasonic_5states[0, 3:15]=np.array(previous_state_3ultrasonic_5states[0, 0:12])
            previous_state_3ultrasonic_5states[0, 0:3] = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            #print("Sensor DP: ")
            #print(sensor_DP1)

            previous_state_DP1[0, 80:640]=np.array(previous_state_DP1[0,0:560])
            previous_state_DP1[0, 0:80]=np.array(sensor_DP1)

            previous_depth_image=image_DP1

            previous_depth_image_5states[1:4]=previous_depth_image_5states[0:3]
            previous_depth_image_5states[0]=image_DP1.reshape(1,80,60)

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
            #action = make_move_ANN_DepthP1(previous_depth_image_5states, model, epsilon)
            action, q_values = make_move_ANN_DepthP1(previous_state_DP1, model, epsilon, distanceCenter, distanceLeft,
                                           distanceRight)

            #print("Q Values: ")
            #print(q_values)
            #time.sleep(10)
            #print("Before training...")


            rospy.sleep(ros_sleep_time)


            #action=0
            #print("Action: ", action)

            current_state_3ultrasonic = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            current_state_3ultrasonic_5states[0, 3:15]=np.array(current_state_3ultrasonic_5states[0, 0:12])
            current_state_3ultrasonic_5states[0, 0:3] = np.array((float(distanceLeft)/1, float(distanceCenter)/1, float(distanceRight)/1)).reshape(1, 3)

            current_state_DP1[0, 80:640]=np.array(current_state_DP1[0,0:560])
            current_state_DP1[0, 0:80]=np.array(sensor_DP1)

            current_depth_image = image_DP1

            current_depth_image_5states[1:4]=current_depth_image_5states[0:3]
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

            #update_Q_value_ANN_DepthP1(previous_depth_image_5states, action, current_depth_image_5states, reward, model,
            #                           epsilon,N_Step_On,Alpha,Lambda, Gamma, Reply_Memory_On, Reply_Memory_New_On, Reply_Memory_New_On_Update)

            if (1==1):#(EveryHowmanyFrames_Counter == EveryHowmanyFrames):
                update_Q_value_ANN_DepthP1(previous_state_DP1, action, current_state_DP1, reward, model,
                                       epsilon,N_Step_On,Alpha,Lambda, Gamma, Reply_Memory_On, Reply_Memory_New_On, Reply_Memory_New_On_Update)
                EveryHowmanyFrames_Counter = 0
            else:
                EveryHowmanyFrames_Counter = EveryHowmanyFrames_Counter + 1

            action_all[i] = action
            reward_all[i] = reward
            q_values_all[i]=q_values.reshape(3)
            #print("Q Values: ")
            #print(q_values_all[i])
            q_value[i] = q_values.reshape(3)[action]

            ''' in order to increase the speed
            previous_depth_image_all[i] = previous_depth_image.reshape(1, 80, 60)
            previous_depth_image_5states_all[i] = previous_depth_image_5states.reshape(4, 80, 60)
            previous_state_3ultrasonic_all[i] = previous_state_3ultrasonic
            previous_state_3ultrasonic_5states_all[i]=previous_state_3ultrasonic_5states



            current_depth_image_all[i] = current_depth_image.reshape(1,80,60)
            current_depth_image_5states_all[i] = current_depth_image_5states.reshape(4, 80, 60)
            current_state_3ultrasonic_all[i] = current_state_3ultrasonic
            current_state_3ultrasonic_5states_all[i] = current_state_3ultrasonic_5states
            '''

            #print("")
            print("")

            i = i + 1

            #runningMean = np.mean(av[:, 1])
            running_reward_mean=running_reward/i

            if(i % 1000==0):
                start_time = time.time()
                plt.plot(i, running_reward_mean, marker='.', color='b')
                #, linewidth=0.2)
                plt.pause(0.05)
                print("is plotting....")
                print("--- plotting --- %s seconds ---" % (time.time() - start_time))

            print("--- Whole --- %s seconds ---" % (time.time() - start_time_whole))

        #plt.show()


        '''

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

        print("training finished.")

        plt.savefig(filename_str + '.jpg')

        print("Saved graph to disk")

        # serialize model to JSON
        model_json = model.to_json()
        with open(filename_str + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(filename_str + ".h5")
        print("Saved model to disk")


        np.savetxt(filename_str + '-data-action.out', action_all)
        np.savetxt(filename_str + '-data-reward.out', reward_all)
        np.savetxt(filename_str + '-data-action-values.out', q_values_all)
        np.savetxt(filename_str + '-data-selected-action-value.out', q_value)

        print("Saved data to disk")

if __name__ == '__main__':
    try:
	init()
        start()
    except rospy.ROSInterruptException:
        pass
