import unittest, pytest, time
import cv2
from Gesture_recognition import hand_gestures
from Hand_tracking import handdetec


#Test 1. Make sure that the camera is opened correctly in a timely manner
@pytest.fixture(scope="module")
def capture():
    #should open the camera
    cap = cv2.VideoCapture(0)
    yield cap.release()

#Test 2. Test different values to make sure open and close functions well
#test that the camera turns on correctly
def test_camera_time(capture):
    start_time = time.time()
    #try to read teh data from the camera
    success, _ = capture.read()
    time_gone = time.time() - start_time
    #check if teh camera has pened correctly
    assert success, "Error: Failed to open the camera"
    #then check if the camera has opened within the last 2 seconds
    #if time is taking very long output that
    assert time_gone < 2.0, "Camera is taking long to open"
#test the different landmarks to verify if the hand is open or closed
@pytest.mark.parametrize("hand_landmarks, expected_gesture",[
    ([[0, 0, 0], [10, 20, 0], [20, 30, 0], [30, 40, 0], [40, 50, 50], [50, 60, 60]], "Open"),
    ([[0, 0, 0], [10, 20, 0], [20, 30, 0], [30, 40, 0], [40, 50, 0], [50, 60, 0]], "Closed"),
])

#test the open close gesture function
def test_open_close_function(hand_landmarks, expected_gesture):
    #call the hand_gesture function
    gesture = hand_gestures(hand_landmarks)
    #want to check if the detected gesture is one of the expected
    assert gesture == expected_gesture, f"Expected: {expected_gesture}, Get: {gesture}"