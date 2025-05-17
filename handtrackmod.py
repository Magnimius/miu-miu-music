import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils 

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
            #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMs, self.mpHands.HAND_CONNECTIONS)

        return img
    def findPosition(self, img, handNo=0, draw = False):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lmk in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lmk.x*w), int(lmk.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255),cv2.FILLED)
        return lmList



def main(): 
    cap = cv2.VideoCapture(0)
    
    success, img = cap.read()

    currentTime = 0
    previousTime = 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        img = cv2.flip(img, 1)
        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,0,0),3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    main()