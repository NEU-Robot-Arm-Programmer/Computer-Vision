#!/usr/bin/python3

import mediapipe as mp
import time
import re
import rospy
import utm
import serial
import threading
from robot_arm.msg import arm_dists

# send initializer message

def serial_send(arg):
    # write the code the arduino needs
    print(arg)

def listener():
    # Arduino port
    #ser_port = rospy.get_param('port2')
    #ser = serial.Serial(ser_port, 9600)

    rospy.init_node('arduino', anonymous=True)
    pub = rospy.Subscriber('/arm_dists', arm_dists, serial_send) 

    rospy.spin()

if __name__ == "__main__":
    listener()


