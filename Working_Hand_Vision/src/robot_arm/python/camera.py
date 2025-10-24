#!/usr/bin/python3

import cv2
import mediapipe as mp
import time
import re
import rospy
import numpy as np
import serial
import threading
import utm
from robot_arm.msg import camera_dists

class handdetec:
    def __init__(self, mode=False, MaxHands=2, modelComplex=1, detection=0.75, tracking=0.75):
        self.mode = mode
        self.MaxHands = MaxHands
        self.complex = modelComplex
        self.detection = detection
        self.track = tracking

        # detecting the landmarks
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.MaxHands, modelComplex, self.detection, tracking)
        self.mpdraw = mp.solutions.drawing_utils

    # job is to find the hands
    def find_hands(self, img, draw=True):
        """Identifies what the hand is in front of teh camera then draws points on it"""
        self.RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(self.RGBimg)
        angles = []

        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                if draw:
                    thumb_tip = (int(hand_landmarks.landmark[4].x * img.shape[1]), int(hand_landmarks.landmark[4].y * img.shape[0]))
                    fingers_center = (int((hand_landmarks.landmark[8].x + hand_landmarks.landmark[12].x +
                                           hand_landmarks.landmark[16].x + hand_landmarks.landmark[20].x) / 4 *
                                          img.shape[1]),
                                      int((hand_landmarks.landmark[8].y + hand_landmarks.landmark[12].y +
                                           hand_landmarks.landmark[16].y + hand_landmarks.landmark[20].y) / 4 *
                                          img.shape[0]))
                    wrist = (int(hand_landmarks.landmark[0].x * img.shape[1]), int(hand_landmarks.landmark[0].y * img.shape[0]))
                    #landmark for the wrist

                    #call the angles function
                    angles = self.calculate_angles(hand_landmarks)
                    for i, angle in enumerate(angles.values()):
                        #check if the angle is a float
                        if isinstance(angle, float):
                            # draw angle near the wrist
                            cv2.putText(img, f'Angles {i + 1}: {angle:.2f}',
                                        (10, img.shape[0] - 20 * (len(angles) - i)), #this places the text under fps
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 255), 1)

                    #Draws the points on fingers, thumb, and wrisr
                    cv2.circle(img, thumb_tip, 20, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, fingers_center, 25, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, wrist, 20, (0, 0, 255), cv2.FILLED)

                    #draws lines to connect the points
                    cv2.line(img, wrist, thumb_tip, (255, 255, 255), 3)  # Line from wrist to thumb
                    cv2.line(img, wrist, fingers_center, (255, 255, 255), 3)  # Line from wrist to fingers center
                    cv2.line(img, thumb_tip, fingers_center, (255, 255, 255), 3)  # Line from thumb to fingers center
        
        return img, angles


    # find the positions (we'll use to then output)
    # might need to change origin_ variables to the previous outputed coordinates instead of 0,0,0
    def find_position(self, img, handNum=0):
        """Finds the position of the hand from the points on the hand"""
        lmlist = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]

            #coordinates fo the origin
            origin_x, origin_y, origin_z = 0.0, 0.0, 0.0

            physical_width = 1.0 #example in meters
            physical_height = 1.0
            physical_depth = 1.0 #this will be used for the z axis
            resolution_width, resolution_height, resolution_depth = img.shape[0], img.shape[1], img.shape[2]

            for id, lm in enumerate(myHand.landmark):
                if id in [0, 4, 8, 12, 16, 20]:
                    cx, cy, cz = int(lm.x * img.shape[1]), int(lm.y * img.shape[0]), 0  # assuming depth is 0

                    real_world_x, real_world_y, real_world_z = pixels_to_meters(cx, cy, cz, physical_width,
                                                                                physical_height, resolution_width,
                                                                                resolution_height, resolution_depth)

                    relative_x = real_world_x - origin_x
                    # these are from debugging print(relative_x)
                    relative_y = real_world_y - origin_y
                    #print(relative_y)
                    relative_z = real_world_z - origin_z
                    #print(relative_z)
                    lmlist.append([id, relative_x, relative_y, relative_z])

        return lmlist

    def calculate_angles(self, hand_landmarks):
        """Calculates the angles that the hand is moving on camera"""
        angles = {}
        # Thumb, index, middle, ring, pinky
        finger_indicies = [4, 8, 12, 16, 20]
        # from teh wrist to the joints of the fingers
        palm_indicies = [0, 1, 5, 9, 13, 17]

        #calc angles between palm and finger??
        for finger_indx in range(len(finger_indicies)):
            #take all te finger numbers
            tip = finger_indicies[finger_indx]
            base = palm_indicies[finger_indx]

            tip_landmark = hand_landmarks.landmark[tip]
            base_landmark = hand_landmarks.landmark[base]
            wrist_landmark = hand_landmarks.landmark[0]

            tip_vector = np.array([tip_landmark.x, tip_landmark.y, tip_landmark.z])
            base_vector = np.array([base_landmark.x, base_landmark.y, base_landmark.z])
            wrist_vector = np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z])

            #calculate the angle between the finger vectors and the wrists
            finger_wrist_vector = wrist_vector - base_vector
            dot_product = np.dot(tip_vector - base_vector, finger_wrist_vector)
            magnitude_prod = np.linalg.norm(tip_vector - base_vector) * np.linalg.norm(finger_wrist_vector)

            if magnitude_prod != 0:
                cos_ang = dot_product / magnitude_prod
                #need to avoid invalid input
                cos_ang = max(min(cos_ang, 1), -1)
                angle = np.arccos(cos_ang)
                angles[f'Finger_{finger_indx} + 1'] = np.degrees(angle)
            else:
                angles[f'Finger_{finger_indx + 1}'] = np.nan

        #calculate the angle of the wrist flex/extend
        wrist_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
        wrist_base = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])

        wrist_vector = wrist_tip - wrist_base
        # Assuming z-axis is upward
        reference_vector = np.array([0, 0, 1])
        wrist_angle = np.arccos(np.dot(wrist_vector, reference_vector) / (
                    np.linalg.norm(wrist_vector) * np.linalg.norm(reference_vector)))
        angles['Wrist_Flexion_Extension'] = np.degrees(wrist_angle)

        return angles

def pixels_to_meters(pixel_x, pixel_y, pixel_z, physical_width, physical_height,
                     resolution_width, resolution_height, resolution_depth):
    """ This converts the pixel output into meters"""
    pixel_size_horizontal = physical_width / resolution_width
    pixel_size_veritcal = physical_height / resolution_height
    pixel_size_depth = physical_width / resolution_depth

    real_world_x = pixel_x * pixel_size_horizontal
    real_world_y = pixel_y * pixel_size_veritcal
    real_world_z = pixel_z * pixel_size_depth

    return real_world_x, real_world_y, real_world_z


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(2)
    detector = handdetec()

    pub = rospy.Publisher('/camera_dists', camera_dists, queue_size=10)
    rospy.init_node('camera', anonymous=True)
    message = camera_dists()

    while True:
        success, img = cap.read()
        # class object
        detector = handdetec()
        img, hand_angle = detector.find_hands(img)
        lm_list = detector.find_position(img)
       # print(lm_list)
        #make sure that the list of coordinates is not

        if not lm_list == [] and not hand_angle == []:
            message.x_dist = lm_list[0][1] * 1000 # convert to millimeters
            message.y_dist = lm_list[0][2] * 1000
            message.z_dist = lm_list[0][3] * 1000
            message.r = hand_angle["Finger_1 + 1"]
            message.p = 0 #hand_angle["Wrist_Flexion_Extension"]
            message.y = 0  
            message.isClosed = False;
            #print(message.z_dist)
            pub.publish(message)
        
        # calculating time
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 0, 255), 1)

        cv2.imshow('Hand Tracking', img) #Window name
        if cv2.waitKey(20) & 0xFF == ord('q'):  #this will close the window
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
