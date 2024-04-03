#!/usr/bin/python3

from robot_arm_IK import inverse_kinematics
import rospy
import time
from robot_arm.msg import arm_dists, camera_dists

pub = rospy.Publisher("/arm_dists", arm_dists)

def listener():
    rospy.init_node('convert_node', anonymous=True)
    convert_sub = rospy.Subscriber("/camera_dists", camera_dists, callback)
    rospy.spin()

def callback(data):
    arm_array = inverse_kinematics(data.x_dist, data.y_dist, data.z_dist, data.r, data.p, data.y)
    r = rospy.Rate(2)

    print(arm_array)
    message = arm_dists()
    message.one = arm_array[0]
    message.two = arm_array[1]
    message.three = arm_array[2]
    message.four = arm_array[3]
    message.five = arm_array[4]
    message.isClosed = data.isClosed

    pub.publish(message)
    r.sleep()

if __name__ == "__main__":
    listener()
