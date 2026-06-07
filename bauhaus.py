import cv2
import numpy as np
import handtrackmod as htm
import time
import threading
import math

FILTERS = ['halftone', 'bitmap', 'ascii', 'glitch', 'cosmos']
current_filter=0
last_wink_time = 0
WINK_COOLDOWN = 1

stars = []
star_colors = [(239,239,239), (194,181,255), (74, 217,234)]

# -- STAR CLASS

class Star:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = np.random.uniform(20, 25)
        self.vy = np.random.uniform(-20, 20)
        self.radius = np.random.randint(40,43)
        index = np.random.randint(0, 3)
        self.colorbgr = star_colors[index]
        self.twinkle_phase = math.radians(np.random.uniform(0,360))
        self.alpha = 255
    
    def update(self):
        self.x +=self.vx
        self.y +=self.vy
        self.twinkle_phase+=math.radians(5)
        self.alpha = int((np.sin(self.twinkle_phase) + 1) / 2 * 255)

    def is_alive(self, max_x, mask):
        if self.x>max_x:
            return False
        if mask[int(self.y), int(self.x)] == 0:
            return False
        return True
    
    def star_points(self, cx, cy, outer_r, inner_r, num_points=5):
        points = []
        for i in range(num_points * 2):
            angle = math.radians(i * 180 / num_points - 90)
            r = outer_r if i % 2 == 0 else inner_r
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            points.append([x, y])
        return np.array(points, dtype=np.int32)
    
    def draw(self, img):
        overlay = img.copy()
        pts = self.star_points(self.x, self.y, self.radius * 3, self.radius)
        cv2.fillPoly(overlay, [pts], self.colorbgr)
        alpha = self.alpha / 255.0
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    

# -- APPLY FILTERS
def apply_filter(img, hull, mode, left_pts=None, right_pts = None):
    if mode == 'halftone':
        img = apply_halftone(img, hull)
    if mode == 'bitmap':
        img = apply_bitmap(img, hull)
    if mode == 'ascii':
        img = apply_ascii(img, hull)
    if mode == 'glitch':
        img = apply_glitch(img, hull)
    if mode == 'cosmos':
        img = apply_cosmos(img, hull, left_pts, right_pts)   
    return img

# -- HULL CREATION LOGIC 

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
        cv2.line(img, p1, p2, color=(0, 0, 255), thickness=2)
    return img

# -- APPLY FILTER LOGIC

def apply_halftone(img, hull, grid_size=10):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    halftone_layer = np.zeros(img.shape, dtype=np.uint8)
    halftone_layer[:] = [139, 90, 20]

    for y in range(0, img.shape[0], grid_size):
        for x in range(0, img.shape[1], grid_size):
            if mask[y, x] > 0:
                brightness = grey_img[y, x]
                max_radius = grid_size//2
                dot_radius = int((255-brightness) / 255*max_radius)
                cv2.circle(halftone_layer, (x, y), dot_radius, (18, 120, 30), -1)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = np.where(mask_3ch > 0, halftone_layer, img)
    return img

def apply_bitmap(img, hull, block_size=15):
    palette = [
        [139, 90, 20],   # cyanotype blue
        [80, 20, 180],   # purple
        [20, 180, 120],  # teal
    ]

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull],255)
    output = img.copy()
    h, w = img.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y+block_size, h)
            x_end = min(x+block_size, w)
            center_x = x+(x_end-x)//2
            center_y = y+(y_end-y)//2

            if mask[center_y, center_x] == 255:
                block = img[y:y_end, x:x_end]
                avg_color = block.mean(axis=(0,1)).astype(int)
                if np.random.random() < 0.05:
                    avg_color = palette[np.random.randint(len(palette))]

                cv2.rectangle(output, (x,y), (x_end, y_end), tuple(map(int, avg_color)),-1)
    np.copyto(img, output, where=(mask[:, :, None]==255))
    return img

def apply_ascii(img, hull, grid_size=12):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull],255)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = np.full_like(img, fill_value=255, dtype=np.uint8)
    palette = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.', ' ']
    h, w = output.shape[:2]
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            if mask[y, x] > 0:
                brightness = grey_img[y, x]
                idx = int(np.interp(brightness, [0, 255], [0, len(palette)-1]))
                cv2.putText(output, palette[idx], (x, y + grid_size), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0),2)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = np.where(mask_3ch > 0, output, img)
    return img

def apply_glitch(img, hull, intensity=20):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    b,g,r = cv2.split(img)
    b = np.roll(b, np.random.randint(-intensity, intensity), axis=1)
    g = np.roll(g, np.random.randint(-intensity, intensity), axis=1)
    r = np.roll(r, np.random.randint(-intensity, intensity), axis=1)
    output = cv2.merge([b, g, r])
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = np.where(mask_3ch > 0, output, img)
    return img

def apply_cosmos(img, hull, left_pts, right_pts):
    overlay = np.zeros_like(img)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    overlay[mask > 0] = [20, 10, 10]
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    left_x = np.min(left_pts[:, 0])
    right_x = np.max(right_pts[:,0])
    valid_ys = np.where(mask[:, int(left_x)] > 0)[0]
    if len(valid_ys) >0 and len(stars) < 150:
        for _ in range(2):
            y = int(np.random.choice(valid_ys))
            stars.append(Star(int(left_x), y))
    for star in stars:
        star.update()
        star.draw(img)
    stars[:] = [s for s in stars if s.is_alive(right_x, mask)]
    return img

# -- FACE DETECTION THREAD MANAGER (threaded for speed optimizing)
   
class FaceDetectorThread:
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.img = None
        self.wink = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            with self.lock:
                img = self.img
            if img is not None:
                self.face_detector.findFace(img)
                wink = self.face_detector.detectWink(img, debug=False)
                with self.lock:
                    self.wink = wink

    def update(self, img):
        with self.lock:
            self.img = img.copy()

    def get_wink(self):
        with self.lock:
            wink = self.wink
            self.wink = None  # consume it
        return wink

    def stop(self):
        self.running = False


# -- MAIN FN LOGIC
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
face_d = htm.faceDetector()
face_thread = FaceDetectorThread(face_d)

while True:
    success, img = cap.read()   
    if not success or img is None:
        continue
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    hands = detector.findPositionBothHands(img)

    face_thread.update(img)
    wink = face_thread.get_wink()
    if wink and (time.time() - last_wink_time) > WINK_COOLDOWN:
        current_filter = (current_filter + 1) % len(FILTERS)
        last_wink_time = time.time()

    left = hands['Left']
    right = hands['Right']

    if len(left) > 0 and len(right) > 0:
        left_points = get_fingertip_points(left)
        right_points = get_fingertip_points(right)
        hull = get_prism_hull(left_points, right_points)
        if current_filter == 4:
            img = apply_filter(img, hull, 'cosmos', left_points, right_points)
        else:
            img = apply_filter(img, hull, FILTERS[current_filter])
        #img = draw_prism_edges(img, left_points, right_points)

    cv2.imshow("Arthouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
