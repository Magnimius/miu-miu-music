import cv2
import numpy as np
import handtrackmod as htm

def get_fingertip_points(lmList) -> np.ndarray:
    numpy_lm_list = np.array(lmList)
    return numpy_lm_list[[4, 8, 12, 16, 20], 1:3]

def get_convex_hull(fingertips) -> np.ndarray:
    fingertips = fingertips.astype(np.int32)
    return cv2.convexHull(points=fingertips)

def get_prism_hull(left_points, right_points):
    prism_hull = np.vstack((left_points, right_points))
    return get_convex_hull(prism_hull)
    
def draw_prism(img, left_lmList, right_lmList, alpha=0.3):
    left_points = get_fingertip_points(left_lmList)
    right_points = get_fingertip_points(right_lmList)
    overlay = img.copy()
    hull = get_prism_hull(left_points, right_points)
    left_hull = get_convex_hull(left_points)
    right_hull = get_convex_hull(right_points)
    cv2.fillPoly(overlay, [hull], color=(255, 255, 255))      # body - white
    cv2.fillPoly(overlay, [left_hull], color=(255, 0, 0))     # left face - blue
    cv2.fillPoly(overlay, [right_hull], color=(0, 0, 255))    # right face - red
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    return img

def draw_prism_edges(img, left_points, right_points):
    for p1, p2 in zip(left_points, right_points):
        p1 = tuple(p1.astype(int))
        p2 = tuple(p2.astype(int))
        cv2.line(img, p1, p2, color=(0, 255, 0), thickness=2)
    return img


cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    hands = detector.findPositionBothHands(img)

    left = hands['Left']
    right = hands['Right']

    if len(left) > 0 and len(right) > 0:
        left_points = get_fingertip_points(left)
        right_points = get_fingertip_points(right)
        img = draw_prism(img, left, right)
        img = draw_prism_edges(img, left_points, right_points)

    cv2.imshow("Arthouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
