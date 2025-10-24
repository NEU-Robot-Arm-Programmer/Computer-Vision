import cv2
import time
from Hand_tracking import handdetec

import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Threshold for hand
OPEN_THRESH = 0.5
CLOSED_THRESH = 150  # or for x it is 0.1 to 0.3 and y 0.4 to 0.7
# Intialize hand tracking
hand = handdetec()
cap = cv2.VideoCapture(0)

""" Based off the movement of the hand it will detect when it opens
    and when it closes and will say that on teh screen
    @param hand_landmarks: the hand coodinates that another function produces
    @return: Text on screen signifying the gesture
    """
def hand_gestures(hand_landmarks):

    thumb_x = hand_landmarks[4][1]
    thumb_y = hand_landmarks[4][2]
    fingers_y = hand_landmarks[5][2]  # weird
    wrist_x = hand_landmarks[0][1]
    wrist_y = hand_landmarks[0][2]

    distance = ((thumb_x - wrist_x) ** 2 + (thumb_y - wrist_y) ** 2) ** 0.5  # this may not be needed for right now

    # Checks if the distance from your fingers and your thumb large
    if thumb_y > fingers_y:  #
        return "Open"
    elif thumb_y < fingers_y:
        return "Closed"
    else:
        return "unknown"


def main():
    cTime = 0
    pTime = 0
    # want to get the landmarks coordinates then call the gesture function

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read")
            break

        img = hand.find_hands(img)
        hand_landmarks = hand.find_position(img)

        if hand_landmarks:
            if len(hand_landmarks) >= 2:  # this is the number of handmarks on teh hand this one is also weird
                gesture = hand_gestures(hand_landmarks)
                print("Detected Gesture:", gesture)

                # display the detected gesture on teh window with the footage
                cv2.putText(img, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                # the camera can't see you hand clearly and it needs to be in sight on window
                print("Not enough hand seen")  # wouldn't this then always output not enough because the code draws 3???

        # display the frame
        cv2.imshow("Hand Tracking", img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FFPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 0, 255), 1)
        # Checks for the key 'q' to quit and closes the window
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    # release rhe camera and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()