import cv2
import handtrackmod as htm

cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    hands = detector.findPositionBothHands(img)
    if hands['Left']:
        print("Left hand detected")
    if hands['Right']:
        print("Right hand detected")
    cv2.imshow("Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break