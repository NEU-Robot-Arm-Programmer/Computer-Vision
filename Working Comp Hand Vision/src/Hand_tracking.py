import adafruit_bno055
import board
import busio
from collections import deque
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import serial

import time
from realsense import depth_frame
from sympy import limit
from adafruit_bno055 import BNO055


# TODO: Be able to calculate the angle that the wrist (marking 0) is bending and rotating at based on the face
#  of the hand Level: Hard
# TODO: Add Rotation and bending of the hand : Level Moderate
# TODO: Have a inverse kinematics file that outputs the movement value in degrees: Level Easy
# TODO: Create code to accept 5 values (the rotations in degrees from inverse kinematics from Serial communication
#  9600 baud Level Moderate -> Hard?
# TODO: IMU + RealSemse z-cords for the hand thing

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

#  initialize configurations
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 64)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 64)
pipeline.start(config)
# Threshold for hand
OPEN_THRESH = 0.5
CLOSED_THRESH = 150  # or for x it is 0.1 to 0.3 and y 0.4 to 0.7
# Reference for real world distance
REFERENCE_DISTANCE = 20 #inches
REFERENCE_Z = 1.0


class handDetec:
    """
    A class for detecting hands to send coordinates to a Robotic Arm.
    """
    def __init__(self, mode=False, maxHands=2, modelComplex=1, detection=0.75, tracking=0.75):
        self.mode = mode
        self.MaxHands = maxHands  # holds the number of hands that the camera will recognize
        self.complex = modelComplex
        self.detection = detection
        self.track = tracking  # will keep track of the speed

        #  detecting the landmarks
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            max_num_hands =self.MaxHands,
            model_complexity=modelComplex,
            min_detection_confidence=self.detection,
            min_tracking_confidence=tracking)
        self.mpdraw = mp.solutions.drawing_utils  # module that draws the points on to the hand
        self.bno = adafruit_bno055.BNO055()
        if not self.bno.begin():
            raise RuntimeError('BNO055 initialization failed')


    def findHands(self, img, draw = True):
        """
        Finds the hand in the camera and draws once found.
        :param img: Image in which hands are detected
        :param draw: Boolean flag to draw landmarks on the hands
        :return: Processed image with landmarks on the hands.
        """
        """ Will find the hand in the camera"""
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.result = self.hands.process(self.imgRGB)

            if self.result.multi_hand_landmarks:
                for handlms in self.result.multi_hand_landmarks:
                    self.mpdraw.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)
                    if draw:
                        # draw only the thumb (landmark 4)
                        self.drawKeyPoints(img, handlms)
            return img #, self.result
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error in processing the hand", e)
            return img, self.result

    def findPosition(self, result, img, handNum=0, draw=True):
        """
        Determine the position of the hand landmarks
        :param result: MediaPipe hand detection result
        :param img: Image containing the hand
        :param handNum: Index of the hand to analyze
        :param draw: Boolean flag to draw landmarks on the hand
        :return: A list of landmark positions to send to an Arduino board.
        """

        #  find the positions (we'll use to then output)
        self.lmlist = []

        if result and result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]

            #  set the origin coordinates
            origin_x, origin_y, origin_z = 0.0, 0.0, 0.0

            physical_width = 1.0  # example in meters
            physical_height = 1.0
            physical_depth = 1.0  # this will be used for the z axis
            resolution_width, resolution_height = img.shape[:2]
            resolution_depth = 1 # place holder for teh camera

            for id, lm in enumerate(myHand.landmark):
                if id in [0, 4, 8, 12, 16, 20]:
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cz = depth_frame.get_distance(cx, cy)
                    # convert the pixels to meters and returns the list
                    real_world_x, real_world_y, real_world_z = pixelsToMeters(cx, cy, cz, physical_width,
                                                                              physical_height, resolution_width,
                                                                              resolution_height, resolution_depth)

                    relative_x = real_world_x - origin_x
                    #  these are from debugging print(relative_x)
                    relative_y = real_world_y - origin_y
                    relative_z = real_world_z - origin_z
                    self.lmlist.append([id, relative_x, relative_y, relative_z])
        return self.lmlist

    def handOrientation(self, lm_list, handedness):
        """
        Identifies whether the palm (palmer side) or the back (dorsal side) of the hand if facing the camera.
        :param lm_list: List of hand landmark positions
        :param handedness: Classifies the result for the orientation
        :return: String indicating the Palm or the Back of a hand.
        """
        if len(lm_list) < 21:
            return "Unknown"

        thumb_tip = next((p for p in lm_list if p[0] == 4), None)
        pinky_tip = next((p for p in lm_list if p[0] == 20), None)
        # Return either Left or Right
        label = handedness.classification[0].label

        print(f"Label: {label}, Thumb: {thumb_tip[1]}, Pinky: {pinky_tip[1]}")

        if thumb_tip is None or pinky_tip is None:
            return "Unknown"

        #For the left hand the thumb is on the right
        if label == "Left":
            if thumb_tip[1] > pinky_tip[1]:
                return "Palm"
            else:
                return "Back"

        else:
            if thumb_tip[0] < pinky_tip[0]:
                return "Palm"
            else:
                return "Back"

    @staticmethod
    def calculateAngles(hand_landmarks):
        """
        Calculates the angles formed by the hand while moving on camera
        :param hand_landmarks: Hand landmarks position
        :return: Dictionary of computed angles
        """
        angles = {}
        # Thumb, index, middle, ring, pinky
        finger_indices = [4, 8, 12, 16, 20]
        # from teh wrist to the joints of the fingers
        palm_indices = [0, 1, 5, 9, 13, 17]

        # calc angles between palm and finger??
        for i in range(len(finger_indices)):
            # take all te finger numbers
            tip = finger_indices[i]
            base = palm_indices[i]

            # place them in variables that will be used in the vectors arrays
            tip_landmark = hand_landmarks.landmark[tip]
            base_landmark = hand_landmarks.landmark[base]
            wrist_landmark = hand_landmarks.landmark[0]

            tip_vector = np.array([tip_landmark.x, tip_landmark.y, tip_landmark.z])
            base_vector = np.array([base_landmark.x, base_landmark.y, base_landmark.z])
            wrist_vector = np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z])

            # calculate the angle between the finger vectors and the wrists
            finger_wrist_vector = wrist_vector - base_vector
            # obtain the dot product of the vectors from the palm(base), fingers, and the tip of the fingers
            dot_product = np.dot(tip_vector - base_vector, finger_wrist_vector)
            # compute the magnitude and multiply the normal of that value
            magnitude_prod = np.linalg.norm(tip_vector - base_vector) * np.linalg.norm(finger_wrist_vector)

            if magnitude_prod != 0:
                cos_ang = dot_product / magnitude_prod
                # need to avoid invalid input
                cos_ang = max(min(cos_ang, 1), -1)
                angle = np.degrees(np.arccos(cos_ang))
                angles[f"Finger_{i + 1}"] = round(angle, 3)
            else:
                angles = 0

        # calculate the angle of the wrist flex/extend
        wrist_tip = np.array(
            [hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
        wrist_base = np.array(
            [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])

        wrist_vector = wrist_tip - wrist_base
        #  Assuming z-axis is upward <- this is stupid assume the z axis is point towards me and away from me
        reference_vector = np.array([0, 0, -1])
        # get the dot product
        wrist_angle = np.arccos(np.dot(wrist_vector, reference_vector) /
                                (np.linalg.norm(wrist_vector) * np.linalg.norm(reference_vector)))
        angles['Wrist'] = np.degrees(wrist_angle)

        return angles


    def handGestures(self, lm_list):
        """
        Determine whether the hand is open or closed based on landmark positions.
        :param lm_list: The list of hand coordinates
        :return: Text on screen indicating 'Open' or 'Close'
        """
        if len(lm_list) < 0:
            return "Unknown"

        thumb_x = lm_list[4][1]
        thumb_y = lm_list[4][2]
        fingers_y = lm_list[5][2]
        # wrist_x = lm_list[0][1]
        # wrist_y = lm_list[0][2]

        # Checks if the distance from your fingers and your thumb large
        if thumb_y > fingers_y:
            return "Open"
        elif thumb_y < fingers_y:
            return "Closed"
        else:
            return "unknown"

    @staticmethod
    def drawKeyPoints(img, landmarks):
        """
        Draws key hand landmarks on teh provided image.
        :param img: Image where landmarks will be drawn
        :param landmarks: Hand landmark data from detection.
        :return:
        """
        for point in [4, 8, 20, 9]:  # the thumb, general area of the 4 fingers, and wrist
            x, y = int(landmarks.landmark[point].x * img.shape[1]), int(landmarks.landmark[point].y * img.shape[0])
            # draw a green circle at each key point
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    def wristBendAngles(self, lm_list):
        """
        Calculates teh angle the wrist is bending to using key hand landmarks
        :param lm_list: List of hand landmark positions
        :return: The angle the wrist is bending to.
        """
        def get_point(lm_id):
            for point in lm_list:
                if point[0] == lm_id:
                    return point[1:]  # this would be the x, y, and z
            return None

        pointA = get_point(0)
        pointB = get_point(4)
        pointC = get_point(20)

        # Wrist
        a = np.array(pointA, dtype=np.float32)
        # Index
        b = np.array(pointB, dtype=np.float32)
        # Pinky
        c = np.array(pointC, dtype=np.float32)

        ba = a - b
        bc = c - b

        cosin_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosin_ang, -1, 1.0))
        return np.degrees(angle)



# === Utility Funcitons ===
def getRealDistance(z_value):
    """
    Compute the real-world distance based on depth values.
    :param z_value: Depth values from 3D camera
    :return: Real-world distance in inches.
    """
    return REFERENCE_DISTANCE * (z_value / REFERENCE_Z)

def pixelsToMeters(pixel_x, pixel_y, pixel_z, physical_width, physical_height,
                     resolution_width, resolution_height, resolution_depth):
    """
    Converts pixel coordinates into meter based measurements.
    :param pixel_x: X-coordinate in pixels.
    :param pixel_y: Y-coordinate in pixels.
    :param pixel_z: Z-coordinate in pixels.
    :param physical_width: Width of the physical scene in meters.
    :param physical_height: Height of the physical scene in meters.
    :param resolution_width: Width of the camera resolution in pixels.
    :param resolution_height: Height of the camera resolution in pixels.
    :param resolution_depth: Depth resolution of the camera depth in meters.
    :return: Converted real-world coordinates in meters.
    """
    # a real distance for example is 19 inches away from the camera so round that up to like 20 inches
    # compare the actual distance vs the computed value  and adjust the calibration factor
    # realHandSize = normalHand * scaledFactor(z)
    pixel_size_horizontal = physical_width / resolution_width * 100
    pixel_size_veritcal = physical_height / resolution_height * 100
    pixel_size_depth = physical_width / resolution_depth * 100

    real_world_x = pixel_x * pixel_size_horizontal
    real_world_y = pixel_y * pixel_size_veritcal
    real_world_z = pixel_z * pixel_size_depth

    return real_world_x, real_world_y, real_world_z

def degreesToSteps(angle, steps_per_rev=200, microstepping=16):
    """
    Compute an angle in degrees to motor steps.
    :param angle: Rotation angle in degrees
    :param steps_per_rev: Number of steps for full revolution of a motor.
    :param microstepping: Microstepping factor.
    :return: Number of steps required for the given angle.
    """
    steps_per_degree = (steps_per_rev * microstepping) / 360
    return int(angle * steps_per_degree)


def main():
    angle_history = deque(maxlen=100)
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'ro-')
    ax.set_ylim(-180, 180)

    cap = cv2.VideoCapture(0)
    detector = handDetec()
    pTime = 0

    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to read")
            break

        img, result = detector.findHands(img)
        cv2.imshow("Hand Tracking", img)

        if result and result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                lm_list = detector.findPosition(img, result)

                if lm_list:
                    orientation = detector.handOrientation(lm_list, handedness)
                    wrist_angle = detector.wristBendAngles(lm_list)

                    if wrist_angle is not None:
                        angle_history.append(wrist_angle)
                        print(f"Orientation: {orientation} | Wrist Bend: {wrist_angle:.2f}°")
                        # cv2.putText(img, f"{hand_label} - {orientation}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2)
                        ax.clear()
                        ax.plot(angle_history)
                        ax.set_title("Wrist Bend Angle")
                        ax.set_xlabel("Frames")
                        ax.set_ylabel("Angle")
                        plt.pause(0.01)

                    # Gestures
                    gesture = detector.handGestures(hand_landmarks)
                    print("Detected Gesture:", gesture)
                    # display the gesture
                    cv2.putText(img, f'Gesture: {gesture}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0,128, 128), 2)

                    print (lm_list)

        # time tracking
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,0,255), 1)
        # cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == '__main__':
    main()
