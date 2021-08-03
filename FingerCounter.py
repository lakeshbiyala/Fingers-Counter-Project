import cv2
import mediapipe as mp
import time
import os

import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(min_detection_confidence=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)
        OpenFingers = []
        # FOR THUMB:
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:  # lmlist[id][x][y]
            OpenFingers.append(1)
        else:
            OpenFingers.append(0)

        # Remaining Fingers:
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:  # lmlist[id][x][y]
                OpenFingers.append(1)
            else:
                OpenFingers.append(0)
        # print(OpenFingers)

        count = 0
        for c in OpenFingers:
            count = count + c

        cv2.putText(img, f'Count: {int(count)}', (00, 70),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
