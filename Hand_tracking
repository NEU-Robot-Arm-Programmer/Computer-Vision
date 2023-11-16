import cv2
import mediapipe as mp
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
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        return lmlist

def pixels_to_meters(pixel_x, pixel_y, physical_width, physical_height, resolution_width, resolution_height):
    pixel_size_horizontal = physical_width / resolution_width
    pixel_size_veritcal = physical_height / resolution_height

    real_world_x = pixel_x * pixel_size_horizontal
    real_world_y = pixel_y * pixel_size_veritcal\

    return real_world_x, real_world_y


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
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

        cv2.imshow('img', img)
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

if __name__ == '__main__':
    main()
