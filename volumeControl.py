import cv2
import time
import numpy as np
import math
import handtrackmod as htm
import osascript


wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

currentTime = 0
previousTime = 0

detector = htm.handDetector()
vol = 50
while True:
    success, img = cap.read()

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    img = cv2.flip(img, 1)
    img = detector.findHands(img)

    lmList=detector.findPosition(img)

    if len(lmList)!=0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        cv2.circle(img, (x1, y1), 15, (70,20, 80), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (70,20, 80), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (80,0,150),2)
        cv2.circle(img, (cx, cy), 15, (70,20, 80), cv2.FILLED)

        length = math.hypot(x2-x2, y2-y1)
        #print(length)
        if length<50:
            cv2.circle(img, (cx,cy),15,(80,200,80), cv2.FILLED)
            if vol>0:
                osascript.osascript(f"set volume output volume {vol}")
                vol-=5
            else:
                vol=0
                osascript.osascript(f"set volume output volume {vol}")
        if length > 300:
            cv2.circle(img, (cx,cy),15,(80,200,80), cv2.FILLED)
            if vol<100:
                osascript.osascript(f"set volume output volume {vol}")
                vol+=5
            else:
                vol=100
                osascript.osascript(f"set volume output volume {vol}")
        

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,0,0),2)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break