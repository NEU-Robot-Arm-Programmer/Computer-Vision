import cv2
import cvzone  #this will allow us to communicate with the arduino

cap = cv2.VideoCapture(1)
detector = cvzone.HandDetector(maxHands=1, detectionCon=0.7)
mySerial = cvzone.SerialObjects("COM3", 9600, 1)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.FindPOsition(img)
    if lmList:
        fingers = detector.fingersUp()
        print(fingers)
        mySerial.sendData(fingers) #get the values from the detectors and sending it
    cv2.imshow("Image", img)
    cv2.waitKey(1)