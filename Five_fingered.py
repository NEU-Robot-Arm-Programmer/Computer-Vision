import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2
import time

class handdetec:
    def __init__(self, mode=False, max_hands=2, modelComplex=1, detection=0.5, tracking=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complex = modelComplex
        self.detection = detection
        self.track = tracking

        # detecting the landmarks
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, modelComplex, detection, tracking)
        self.mpdraw = mp.solutions.drawing_utils

    # job is to find the hands
    def findhands(self, img, draw=True):
        self.RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(self.RGBimg)

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

                    angles = self.calculate_angles(handlms)
                    for i, angle in enumerate(angles.values()):
                        # check if the angle is a float
                        if isinstance(angle, float):
                            # draw angle near the wrist
                            cv2.putText(img, f'Angle {i + 1}: {angle:.2f}',
                                        (10, img.shape[0] - 20 * (len(angles) - i)),  # this places the text under fps
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        return img


    # find the positions (we'll use to then output)
    def findposition(self, img, handNum=0, draw=True):
        lmlist = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]

            physical_width = 1.0 #example in meters
            physical_height = 1.0
            resolution_width, resolution_height = img.shape[1], img.shape[0]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  #, int(lm.z * l)
                #convert the pixels to meters
                real_world_x, real_world_y = pixels_to_meters(cx, cy, physical_width, physical_height, resolution_width, resolution_height)

                lmlist.append([id, real_world_x, real_world_y]) #cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return lmlist

    def calculate_angles(self, hand_landmarks):
        """Calculates the angles that the hand is moving on camera"""
        angles = {}

        finger_indicies = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        palm_indicies = [0, 1, 5, 9, 13, 17]  # from teh wrist to the joints of the fingers

        # calc angles between palm and finger??
        for finger_indx in range(len(finger_indicies)):
            tip = finger_indicies[finger_indx]
            base = palm_indicies[finger_indx]

            tip_landmark = hand_landmarks.landmark[tip]
            base_landmark = hand_landmarks.landmark[base]
            wrist_landmark = hand_landmarks.landmark[0]

            tip_vector = np.array([tip_landmark.x, tip_landmark.y, tip_landmark.z])
            base_vector = np.array([base_landmark.x, base_landmark.y, base_landmark.z])
            wrist_vector = np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z])

            # calculate teh angle between teh finger vectors and the wrists
            finger_wrist_vector = wrist_vector - base_vector
            dot_product = np.dot(tip_vector - base_vector, finger_wrist_vector)
            magnitude_prod = np.linalg.norm(tip_vector - base_vector) * np.linalg.norm(finger_wrist_vector)

            if magnitude_prod != 0:
                cos_ang = dot_product / magnitude_prod
                # need to avoid invalid input
                cos_ang = max(min(cos_ang, 1), -1)
                angle = np.arccos(cos_ang)
                angles[f'Finger_{finger_indx} + 1'] = np.degrees(angle)
            else:
                angles[f'Finger_{finger_indx + 1}'] = np.nan

        # calculate the angle of the wrist flex/extend
        wrist_tip = np.array(
            [hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
        wrist_base = np.array(
            [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])

        wrist_vector = wrist_tip - wrist_base
        reference_vector = np.array([0, 0, 1])  # Assuming z-axis is upward
        wrist_angle = np.arccos(np.dot(wrist_vector, reference_vector) / (
                np.linalg.norm(wrist_vector) * np.linalg.norm(reference_vector)))
        angles['Wrist_Flexion_Extension'] = np.degrees(wrist_angle)

        return angles

def pixels_to_meters(pixel_x, pixel_y, physical_width, physical_height, resolution_width, resolution_height):
    pixel_size_horizontal = physical_width / resolution_width
    pixel_size_veritcal = physical_height / resolution_height

    real_world_x = pixel_x * pixel_size_horizontal
    real_world_y = pixel_y * pixel_size_veritcal\

    return real_world_x, real_world_y


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(4)
    while True:
        success, img = cap.read()
        # class object
        detector = handdetec()
        img = detector.findhands(img)
        position = detector.findposition(img)

        if len(position) != 0:
            print(position)

        # calculating time
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        cv2.imshow('Hand Track', img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()

