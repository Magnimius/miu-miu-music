import cv2
import time
import numpy as np
import math
import handtrackmod as htm
import osascript
import threading
from collections import deque


wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

currentTime = 0
previousTime = 0

detector = htm.handDetector()
vol = 50

reverb_level = 0
reverb_history = []
reverb_smooth_frames = 8
reverb_trails = deque(maxlen=20)  # For visual trails

# Visual reverb effect parameters
echo_particles = []

currentTime = 0
previousTime = 0
last_reverb_update = 0
class EchoParticle:
    def __init__(self, x, y, intensity):
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        self.intensity = intensity
        self.age = 0
        self.max_age = int(30 * (intensity / 100))  # Age based on reverb intensity
        self.radius = 5
        
    def update(self):
        self.age += 1
        # Create expanding echo effect
        expansion = (self.age / self.max_age) * 50
        self.x = self.original_x + np.random.randint(-int(expansion), int(expansion))
        self.y = self.original_y + np.random.randint(-int(expansion), int(expansion))
        self.radius = 5 + (self.age / self.max_age) * 15
        
    def is_alive(self):
        return self.age < self.max_age
    
    def draw(self, img):
        alpha = 1 - (self.age / self.max_age)
        color = (int(100 * alpha), int(150 * alpha), int(255 * alpha))
        cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), color, 2)

def smooth_reverb(new_reverb, history, max_frames):
    history.append(new_reverb)
    if len(history) > max_frames:
        history.pop(0)
    return sum(history) / len(history)

def calc_reverb_from_hand_height(lmList):
    """Calculate reverb based on hand height in frame"""
    if len(lmList) < 9:
        return 0
    
    # Use middle finger tip (landmark 12) for height detection
    hand_y = lmList[12][2]
    frame_height = hCam
    
    # Convert Y position to reverb level (higher hand = more reverb)
    # Invert because Y=0 is top of frame
    reverb_percentage = ((frame_height - hand_y) / frame_height) * 100
    
    # Clamp between 0 and 100
    return max(0, min(100, reverb_percentage))

def calc_reverb_from_hand_openness(lmList):
    """Calculate reverb based on how open/closed the hand is"""
    if len(lmList) < 21:
        return 0
    
    # Calculate distances between fingertips and palm center
    palm_x, palm_y = lmList[0][1], lmList[0][2]  # Wrist as palm reference
    
    distances = []
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    
    for tip_idx in fingertips:
        tip_x, tip_y = lmList[tip_idx][1], lmList[tip_idx][2]
        distance = math.hypot(tip_x - palm_x, tip_y - palm_y)
        distances.append(distance)
    
    # Average distance represents hand openness
    avg_distance = sum(distances) / len(distances)
    
    # Convert to reverb level (more open = more reverb)
    # Adjust these values based on your camera setup
    min_distance, max_distance = 50, 200
    reverb_level = ((avg_distance - min_distance) / (max_distance - min_distance)) * 100
    
    return max(0, min(100, reverb_level))

def apply_macos_reverb_effect(reverb_level):
    if reverb_level > 70:
        # Enable audio enhancement (simulates reverb)
        osascript.osascript('''
        tell application "System Preferences"
            set current pane to pane "com.apple.preference.sound"
        end tell
        ''')
    return

def add_visual_reverb_effects(img, hand_positions, reverb_level):
    """Add visual reverb effects to the camera feed"""
    global echo_particles, reverb_trails
    
    if len(hand_positions) > 0:
        # Add hand position to trail
        main_hand_x = hand_positions[9][1]  # Middle finger MCP
        main_hand_y = hand_positions[9][2]
        reverb_trails.append((main_hand_x, main_hand_y))
        
        # Create new echo particles based on reverb level
        if reverb_level > 10 and len(echo_particles) < 50:
            for _ in range(int(reverb_level / 25) + 1):
                particle = EchoParticle(
                    main_hand_x + np.random.randint(-30, 30),
                    main_hand_y + np.random.randint(-30, 30),
                    reverb_level
                )
                echo_particles.append(particle)
    
    # Update and draw echo particles
    echo_particles = [p for p in echo_particles if p.is_alive()]
    for particle in echo_particles:
        particle.update()
        particle.draw(img)
    
    # Draw reverb trails with intensity-based opacity
    if len(reverb_trails) > 1:
        trail_points = list(reverb_trails)
        for i in range(1, len(trail_points)):
            alpha = i / len(trail_points)
            thickness = max(1, int(alpha * (reverb_level / 15)))
            intensity = reverb_level / 100
            color = (
                int(50 + 100 * intensity), 
                int(100 + 100 * intensity), 
                int(200 + 55 * intensity)
            )
            cv2.line(img, trail_points[i-1], trail_points[i], color, thickness)
    
    # Add reverb glow effect around hand
    if reverb_level > 30:
        glow_radius = int(30 + reverb_level / 2)
        glow_intensity = reverb_level / 100
        for pos in hand_positions[::4]:  # Every 4th landmark for performance
            x, y = pos[1], pos[2]
            cv2.circle(img, (x, y), glow_radius, (int(50 * glow_intensity), int(100 * glow_intensity), int(255 * glow_intensity)), 1)
    
    # Draw reverb level visualization
    draw_reverb_meter(img, reverb_level)
    draw_reverb_waveform(img, reverb_level)

def draw_reverb_meter(img, reverb_level):
    """Draw a visual meter showing current reverb level"""
    meter_x, meter_y = 50, 300
    meter_width, meter_height = 30, 200
    
    # Background
    cv2.rectangle(img, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (30, 30, 30), cv2.FILLED)
    cv2.rectangle(img, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (100, 100, 100), 2)
    
    # Reverb level fill
    fill_height = int((reverb_level / 100) * meter_height)
    fill_y = meter_y + meter_height - fill_height
    
    # Color changes based on reverb level
    if reverb_level < 30:
        color = (0, 255, 100)  # Green
    elif reverb_level < 70:
        color = (0, 255, 255)  # Yellow
    else:
        color = (100, 150, 255)  # Orange
    
    if fill_height > 0:
        cv2.rectangle(img, (meter_x, fill_y), (meter_x + meter_width, meter_y + meter_height), color, cv2.FILLED)
    
    # Labels
    cv2.putText(img, "REVERB", (meter_x - 10, meter_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"{int(reverb_level)}%", (meter_x + 40, meter_y + meter_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_reverb_waveform(img, reverb_level):
    """Draw animated waveform visualization"""
    wave_x, wave_y = 120, 350
    wave_width = 300
    wave_height = 80
    
    # Background
    cv2.rectangle(img, (wave_x, wave_y), (wave_x + wave_width, wave_y + wave_height), (20, 20, 20), cv2.FILLED)
    
    # Generate waveform based on reverb level and time
    current_time = time.time()
    points = []
    
    for i in range(0, wave_width, 3):
        # Create multiple sine waves for reverb effect
        base_freq = 0.1
        reverb_freq = 0.05 * (reverb_level / 100)
        
        amplitude = (reverb_level / 100) * (wave_height // 4)
        
        # Main wave
        main_wave = amplitude * math.sin(current_time * 5 + i * base_freq)
        
        # Reverb waves (multiple echoes)
        reverb_wave = 0
        for echo in range(3):
            delay = echo * 0.2
            decay = 0.7 ** echo
            reverb_wave += (amplitude * decay * 
                          math.sin((current_time - delay) * 5 + i * reverb_freq))
        
        total_wave = main_wave + (reverb_wave * reverb_level / 100)
        
        y_pos = wave_y + wave_height//2 + int(total_wave)
        points.append((wave_x + i, y_pos))
    
    # Draw waveform
    if len(points) > 1:
        for i in range(len(points) - 1):
            intensity = reverb_level / 100
            color = (int(100 * intensity), int(200 * intensity), int(255 * intensity))
            cv2.line(img, points[i], points[i + 1], color, 2)


prev_y = 0  # Previous Y position of wrist
flick_threshold = 200  # Minimum vertical movement for flick detection
last_flick_time = 0  # To prevent multiple detections
cooldown = 2  # Time between allowed flicks (seconds)


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

        raw_reverb = calc_reverb_from_hand_height(lmList)
        #raw_reverb = calc_reverb_from_hand_openness(lmList)
        smooth_reverb_value = smooth_reverb(raw_reverb, reverb_history, reverb_smooth_frames)
        
        if abs(smooth_reverb_value - reverb_level) > 5 and (currentTime - last_reverb_update) > 0.5:
            reverb_level = int(smooth_reverb_value)
            apply_macos_reverb_effect(reverb_level)
            last_reverb_update = currentTime

        add_visual_reverb_effects(img, lmList, reverb_level)


        middle_y = lmList[12][2]  # Y-coordinate of wrist
        
        # Calculate vertical movement (negative value means upward movement)
        y_movement = middle_y - prev_y
        
        # Only check for flicks if we have a previous position and cooldown time has passed
        if prev_y != 0 and (currentTime - last_flick_time) > cooldown:
            # Detect upward flick (when hand moves up rapidly)
            if y_movement < -flick_threshold:
                # Upward flick detected!
                print("UPWARD FLICK DETECTED!")
                last_flick_time = currentTime
                flick_direction = "up"
                
                # Set volume to 70 for upward flick
                vol = 70
                osascript.osascript(f"set volume output volume {vol}")
                print("VOLUME SET TO 70!")
                
                # Visual feedback
                cv2.putText(img, "FLICK UP!", (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 3)
            
            # Detect downward flick (when hand moves down rapidly)
            elif y_movement > flick_threshold:
                # Downward flick detected!
                print("DOWNWARD FLICK DETECTED!")
                last_flick_time = currentTime
                flick_direction = "down"
                
                # Set volume to 30 for downward flick
                vol = 30
                osascript.osascript(f"set volume output volume {vol}")
                print("VOLUME SET TO 30!")
                
                # Visual feedback
                cv2.putText(img, "FLICK DOWN!", (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 3)
        
        # Save current position for next frame
        prev_y = middle_y
        

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,0,0),2)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break