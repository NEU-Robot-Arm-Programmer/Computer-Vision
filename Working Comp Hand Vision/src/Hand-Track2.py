import cv2
import mediapipe as mp
import time
from matplotlib import pyplot as plt
import numpy as np


class handdetec:
    def __init__(self, mode=False, max_hands=2, modelComplex=1, detection=1.5, tracking=1.5):
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
    def find_hands(self, img, draw=True):
        """Identifies what the hand is in front of teh camera then draws points on it"""
        self.RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(self.RGBimg)

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

                    #Draws the points on fingers, thumb, and wrisr
                    cv2.circle(img, thumb_tip, 20, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, fingers_center, 25, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, wrist, 20, (0, 0, 255), cv2.FILLED)

                    #draws lines to connect the points
                    cv2.line(img, wrist, thumb_tip, (255, 255, 255), 3)  # Line from wrist to thumb
                    cv2.line(img, wrist, fingers_center, (255, 255, 255), 3)  # Line from wrist to fingers center
                    cv2.line(img, thumb_tip, fingers_center, (255, 255, 255), 3)  # Line from thumb to fingers center
                    # cv2.circle(img, wrist, 10, (0, 128, 128), cv2.FILLED)
                    # points_to_draw = [0, 4, 8, 12, 16, 20]
                    # self.mpdraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS,
                    #                            landmark_drawing_spec=self.mpdraw.DrawingSpec(color=(0, 128, 128),
                    #                                                                          circle_radius=4),
                    #                            connection_drawing_spec=self.mpdraw.DrawingSpec(color=(255, 0, 0),
                    #                                                                            thickness=2))

        return img


    # find the positions (we'll use to then output)
    def find_position(self, img, handNum=0):
        """Finds the position of the hand from the points on the hand"""
        lmlist = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]

            #coordinates fo the origin
            origin_x, origin_y, origin_z = 0.0, 0.0, 0.0

            physical_width = 1.0 #example in meters
            physical_height = 1.0
            physical_depth = 1.0
            resolution_width, resolution_height, resolution_depth = img.shape[0], img.shape[1], img.shape[2]

            for id, lm in enumerate(myHand.landmark):
                if id in [0, 4, 8, 12, 16, 20]:
                    cx, cy, cz = int(lm.x * img.shape[1]), int(lm.y * img.shape[0]), 0  # assuming depth is 0

                    real_world_x, real_world_y, real_world_z = pixels_to_meters(cx, cy, cz, physical_width,
                                                                                physical_height, resolution_width,
                                                                                resolution_height, resolution_depth)

                    relative_x = real_world_x - origin_x
                    relative_y = real_world_y - origin_y
                    relative_z = real_world_z - origin_z
                    lmlist.append([id, relative_x, relative_y, relative_z])

        return lmlist

def pixels_to_meters(pixel_x, pixel_y, pixel_z, physical_width, physical_height, resolution_width, resolution_height, resolution_depth):
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
    cap = cv2.VideoCapture(0)
    detector = handdetec()

    while True:
        success, img = cap.read()
        # class object
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list)

        # calculating time
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FFPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 0, 255), 1)

        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
