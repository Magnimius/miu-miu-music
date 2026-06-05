import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import numpy as np
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

class handDetector():
    def __init__(self, detectionCon=0.7, trackCon=0.7):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.timestamp_ms = 0

    def findHands(self, img, draw=True):
        self.timestamp_ms += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect_for_video(mp_image, self.timestamp_ms)

        if draw and self.results.hand_landmarks:
            h, w, _ = img.shape
            for hand_landmarks in self.results.hand_landmarks:
                # Draw connections
                connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                for connection in connections:
                    start = connection.start
                    end = connection.end
                    x1 = int(hand_landmarks[start].x * w)
                    y1 = int(hand_landmarks[start].y * h)
                    x2 = int(hand_landmarks[end].x * w)
                    y2 = int(hand_landmarks[end].y * h)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 200, 200), 2)
                # Draw landmark dots
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return img

    def findPositionBothHands(self, img):
        """
        Returns dict: {'Left': lmList or [], 'Right': lmList or []}
        Each lmList is [[id, cx, cy], ...]
        """
        hands_data = {'Left': [], 'Right': []}

        if not self.results or not self.results.hand_landmarks:
            return hands_data

        h, w, _ = img.shape
        for hand_landmarks, handedness in zip(self.results.hand_landmarks, self.results.handedness):
            label = handedness[0].category_name
            # Flip label to match mirrored feed
            label = 'Right' if label == 'Left' else 'Left'

            lmList = []
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            hands_data[label] = lmList

        return hands_data

    def findPosition(self, img, handNo=0):
        """Legacy single-hand method"""
        lmList = []
        if not self.results or not self.results.hand_landmarks:
            return lmList
        if handNo < len(self.results.hand_landmarks):
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.hand_landmarks[handNo]):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList