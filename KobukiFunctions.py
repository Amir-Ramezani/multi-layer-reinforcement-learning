

import rospy

from std_msgs.msg import String
from sensor_msgs.msg import Range
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import MotorPower

global velocity_publisher
velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

global cmd
cmd = Twist()
cmd.linear.x = 0.0
cmd.linear.y = 0.0
cmd.linear.z = 0.0
cmd.angular.x = 0.0
cmd.angular.y = 0.0
cmd.angular.z = 0.0

global distanceLeft, distanceCenter, distanceRight
distanceCenter=0
distanceLeft=0
distanceRight=0

################################# Kobuki movement functions

def kobuki_move_forward():
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

        if (distanceCenter > 40):
            if (distanceLeft > 40 and distanceRight > 40):
                velocity_publisher.publish(cmd)

        if (distanceCenter > 40):
            if (distanceLeft > 40 and distanceRight > 40):
                velocity_publisher.publish(cmd)

    else:
        velocity_publisher.publish(cmd)
        velocity_publisher.publish(cmd)
        velocity_publisher.publish(cmd)
    return True

def kobuki_turn_left():
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 1.0

    velocity_publisher.publish(cmd)
    return True

def kobuki_turn_right():
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -1.0
    velocity_publisher.publish(cmd)
    return True

def kobuki_turn_left_slow():
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.6

    velocity_publisher.publish(cmd)
    return True

def kobuki_turn_right_slow():
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -0.6

    velocity_publisher.publish(cmd)
    return True

def kobuki_turn_left_fast():
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 2.6

    velocity_publisher.publish(cmd)
    return True

def kobuki_turn_right_fast():
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -2.6

    velocity_publisher.publish(cmd)
    return True

def kobuki_rotate_right_180():
    cmd.linear.x = 0.0
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = -5.0

    velocity_publisher.publish(cmd)
    return True

def kobuki_move_forward_slow():
    cmd.linear.x = 0.25
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.0

    velocity_publisher.publish(cmd)
    return True

def kobuki_move_backward():
    cmd.linear.x = -0.5
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.0

    velocity_publisher.publish(cmd)
    return True

def move_turtle_bot(turtlebot_action, dC, dL, dR):
    global distanceLeft, distanceCenter, distanceRight
    distanceCenter = dC
    distanceLeft = dL
    distanceRight = dR

    if (turtlebot_action== 0):
        kobuki_move_forward()

    if(turtlebot_action== 1):
        kobuki_turn_left_slow()

    if (turtlebot_action == 2):
        kobuki_turn_right_slow()

    if(turtlebot_action == 3):
        kobuki_turn_left_fast()

    if (turtlebot_action == 4):
        kobuki_turn_right_fast()

    return 0
