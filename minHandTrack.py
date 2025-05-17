import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands() #can change tracking and detection thresholds
mpDraw = mp.solutions.drawing_utils

currentTime = 0
previousTime = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMs in results.multi_hand_landmarks:
            for id, lmk in enumerate(handLMs.landmark):
                h, w, c = img.shape
                cx, cy = int(lmk.x*w), int(lmk.y*h)
            mpDraw.draw_landmarks(img, handLMs, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,0,0),3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break