import time, cv2, pyrealsense2
import mediapipe as mp
import math

#TODO:
# rewrite code to have 2 marks
# measure the distance between the two points (signifies open and close by distance)

class handdetec:
    """ Will identify the hand and then track its movement by printing out
    the x,y, and z coordinates that are in pixels.
    """
    def __init__(self, mode=False, max_hands=1, modelComplex=1, detection=1, tracking=1):
        self.mode = mode
        self.max_hands = max_hands
        # the hands should now be 1, as the claw should only recognize 1 persons hand to move
        self.complex = modelComplex
        self.detection = detection
        self.track = tracking

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, modelComplex, detection,tracking)
        self.mpdraw = mp.solutions.drawing_utils

    def findhand(self, img, draw = True):
        """ Will find the hand in the camera"""
        try:
            self.RBGimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.result = self.hands.process(self.RBGimg)

            if self.result.multi_hand_landmarks:
                for handlms in self.result.multi_hand_landmarks:
                    if draw:
                        # draw only the thumb (landmark 4)
                        self.draw_key_points(img, handlms)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error in processing the hand", e)
            return img

        return img
                    #
                    # thumb_x, thumb_y = int(handlms.landmark[4].x * img.shape[1]), int(handlms.landmark[4].y * img.shape[0])
                    # cv2.circle(img, (thumb_x, thumb_y), 10, (0, 128, 128), cv2.FILLED)
                    #
                    # # calculate the average position of the four fingers
                    # fingers_x = sum(int(handlms.landmark[i].x * img.shape[1]) for i in range(5, 21)) / 16
                    # fingers_y = sum(int(handlms.landmark[i].y * img.shape[0]) for i in range(5, 21)) / 16
                    # cv2.circle(img, (int(fingers_x), int(fingers_y)), 10, (0, 0, 255), cv2.FILLED)

    def draw_key_points(self, img, landmarks):
        for point in [4, 8, 0]: #the thumb, general area of the 4 fingers, and wrist
            x,y = int(landmarks.landmark[point].x * img.shape[1]), int(landmarks.landmark[point].y * img.shape[0])
            #draw a greeen circle at each key point
            cv2.circle(img, (x,y), 5, (0, 255, 0), -1)

    # Find the positions (will be used as output)
    def findposition(self, img, handNum=0, draw=True):
        lmlist = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]

            # coordinates fo the origin
            origin_x, origin_y, origin_z = 0.0, 0.0, 0.0

            physical_width = 1.0  # example in meters
            physical_height = 1.0
            physical_depth = 1.0
            resolution_width, resolution_height, resolution_depth = img.shape[0], img.shape[1], img.shape[2]

            for id, lm in enumerate(myHand.landmark):
                h, w, d = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * d)

                # convert the pixels to meters
                real_world_x, real_world_y, real_world_z = pixels_to_meters(cx, cy, cz,
                                                                            physical_depth, physical_width,
                                                                            physical_height, resolution_width,
                                                                            resolution_height)

                # now make the coordinates relative to the origin
                relative_x = real_world_x - origin_x
                relative_y = real_world_y - origin_y
                relative_z = real_world_z - origin_z
                lmlist.append([id, relative_x, relative_y, relative_z])
                # if draw:
                #     cv2.circle(img, (cx, cy), 10, (0, 128, 128), cv2.FILLED)
        return lmlist


# Converts the pixel values to meters
def pixels_to_meters(pixel_x, pixel_y, pixel_z,
                     physical_width, physical_height, resolution_width,
                     resolution_height, resolution_depth):
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
        # create an object for the hand
        img = detector.findhand(img)
        position = detector.findposition(img)

        if len(position) != 0:
            # Extract coordinates of thumb (id=4)
            thumb_coords = position[4][1:4]  # [x, y, z]
            fingers_avg_coords = position[16][1:4]  # [x, y, z]

            # Calculate the distance between thumb and fingers_avg
            distance = math.sqrt((thumb_coords[0] - fingers_avg_coords[0]) ** 2 +
                                 (thumb_coords[1] - fingers_avg_coords[1]) ** 2 +
                                 (thumb_coords[2] - fingers_avg_coords[2]) ** 2)

            print("Distance between thumb and fingers average:", distance)

        # calculate the time (this may not be needed
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 0, 255), 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()