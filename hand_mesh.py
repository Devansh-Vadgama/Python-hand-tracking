"""
hand_tracker_with_game.py

A hand‐tracking “game” where colored squares and circles fall from the top.
Touch squares to increase your score; touch circles to decrease it.
Squares speed up over time. Track up to two hands with finger detection.

Features:
  • Resolution: 640×480 for speed
  • Up to two hands (MediaPipe 21-point model)
  • Left hand = Blue skeleton, Right hand = Green skeleton
  • Falling squares (purple): touching one → +1 score
  • Falling circles (cyan): touching one → –1 score
  • Squares spawn every 1 second; circles every 5 seconds
  • Square falling speed increases by 1 px/frame every 10 seconds (max cap)
  • Score displayed top-left
  • Finger count displayed near each wrist

Prerequisites (Windows + Python 3.10 64-bit):
    py -3.10 -m venv venv
    .\\venv\\Scripts\\Activate.ps1
    pip install --upgrade pip setuptools wheel
    pip install mediapipe opencv-python numpy

Run:
    python hand_mesh.py
Press ESC to exit.
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time

# -----------------------------------------------------------------------------
# 1) Constants & Helper Functions
# -----------------------------------------------------------------------------
# Color definitions (BGR)
LEFT_HAND_COLOR    = (255, 100, 0)    # Blueish (Left hand)
RIGHT_HAND_COLOR   = (0, 255, 0)      # Green   (Right hand)
FINGERTIP_COLOR    = (0, 0, 255)      # Red     (Fingertips)
FINGER_COUNT_COLOR = (0, 255, 255)    # Yellow  (Finger count text)
SQUARE_COLOR       = (200, 50, 200)   # Purple  (Squares)
CIRCLE_COLOR       = (255, 200, 0)    # Cyan    (Circles)
SCORE_COLOR        = (255, 255, 255)  # White   (Score text)

# Drawing settings
LINE_THICKNESS     = 2    # Skeleton line thickness
LANDMARK_RADIUS    = 4    # Skeleton landmark radius
FINGERTIP_RADIUS   = 6    # Fingertip highlight radius

# Gameplay settings
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480

SQUARE_SIZE        = 40          # px
CIRCLE_RADIUS_GAME = 20          # px
SQUARE_BASE_SPEED  = 4           # px/frame
SPEED_INCREMENT    = 1           # px/frame increase
SPEED_INCREMENT_INTERVAL = 10.0  # sec between speed increases
MAX_SQUARE_SPEED   = 15          # px/frame cap

NEW_SQUARE_INTERVAL  = 1.0        # sec between square spawns
NEW_CIRCLE_INTERVAL  = 5.0        # sec between circle spawns

# Indices for fingertip & PIP joints (21-point model)
FINGER_TIPS = [4, 8, 12, 16, 20]   # Thumb tip=4, Index tip=8, etc.
FINGER_PIPS = [2, 6, 10, 14, 18]   # Thumb MCP=2, Index PIP=6, etc.

def count_extended_fingers(hand_landmarks):
    """
    Count how many fingers are extended (0–5). Uses simple geometry:
      - For Index, Middle, Ring, Pinky: tip.y < pip.y means extended.
      - For Thumb: tip.x < pip.x for left hand, tip.x > pip.x for right hand.
    Returns (count, is_right).
    """
    lm = hand_landmarks.landmark
    count = 0

    # Determine hand label by comparing wrist (0) and index_mcp (5) x-coordinates
    is_right = lm[0].x < lm[5].x

    # Thumb
    tip_x, pip_x = lm[FINGER_TIPS[0]].x, lm[FINGER_PIPS[0]].x
    if (is_right and tip_x > pip_x) or (not is_right and tip_x < pip_x):
        count += 1

    # Other fingers
    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if lm[tip_idx].y < lm[pip_idx].y:
            count += 1

    return count, is_right

def fingertip_positions(hand_landmarks, frame_w, frame_h):
    """
    Returns a list of (x_px, y_px) pixel coords for each fingertip.
    """
    pts = []
    for tip_idx in FINGER_TIPS:
        lm = hand_landmarks.landmark[tip_idx]
        x_px = int(lm.x * frame_w)
        y_px = int(lm.y * frame_h)
        pts.append((x_px, y_px))
    return pts

# -----------------------------------------------------------------------------
# 2) Initialize MediaPipe Hands & OpenCV
# -----------------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,            # 21-point model
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# -----------------------------------------------------------------------------
# 3) Game State Initialization
# -----------------------------------------------------------------------------
squares = []   # list of dicts: {'x': int, 'y': int}
circles = []   # list of dicts: {'x': int, 'y': int}

last_square_time = time.time()
last_circle_time = time.time()
game_start_time  = time.time()
square_speed     = SQUARE_BASE_SPEED
score            = 0

def spawn_square():
    """Spawn a new square at a random x at the top (y=0)."""
    x = random.randint(0, FRAME_WIDTH - SQUARE_SIZE)
    squares.append({'x': x, 'y': 0})

def spawn_circle():
    """Spawn a new circle at a random x at the top (y=0)."""
    x = random.randint(CIRCLE_RADIUS_GAME, FRAME_WIDTH - CIRCLE_RADIUS_GAME)
    circles.append({'x': x, 'y': 0})

def update_falling_objects():
    """Move squares and circles downward; remove off-screen."""
    global squares, circles
    # Update squares
    remove_sq = []
    for i, sq in enumerate(squares):
        sq['y'] += square_speed
        if sq['y'] > FRAME_HEIGHT:
            remove_sq.append(i)
    for i in reversed(remove_sq):
        squares.pop(i)
    # Update circles
    remove_c = []
    for i, cir in enumerate(circles):
        cir['y'] += square_speed  # circles fall at same speed
        if cir['y'] - CIRCLE_RADIUS_GAME > FRAME_HEIGHT:
            remove_c.append(i)
    for i in reversed(remove_c):
        circles.pop(i)

def draw_falling_objects(frame):
    """Draw all squares and circles onto the frame."""
    for sq in squares:
        x, y = sq['x'], sq['y']
        cv2.rectangle(frame, (x, y), (x + SQUARE_SIZE, y + SQUARE_SIZE),
                      SQUARE_COLOR, -1)
    for cir in circles:
        x, y = cir['x'], cir['y']
        cv2.circle(frame, (x, y), CIRCLE_RADIUS_GAME, CIRCLE_COLOR, -1)

def check_collisions(fingertips):
    """Remove squares/circles touched by any fingertip; update score."""
    global squares, circles, score
    # Check squares
    rem_sq = []
    for i, sq in enumerate(squares):
        x, y = sq['x'], sq['y']
        for fx, fy in fingertips:
            if x <= fx <= x + SQUARE_SIZE and y <= fy <= y + SQUARE_SIZE:
                score += 1
                rem_sq.append(i)
                break
    for i in reversed(rem_sq):
        squares.pop(i)
    # Check circles
    rem_c = []
    for i, cir in enumerate(circles):
        x, y = cir['x'], cir['y']
        for fx, fy in fingertips:
            # circle collision: distance from center <= radius
            if (fx - x)**2 + (fy - y)**2 <= CIRCLE_RADIUS_GAME**2:
                score -= 1
                rem_c.append(i)
                break
    for i in reversed(rem_c):
        circles.pop(i)

# -----------------------------------------------------------------------------
# 4) Main Loop: Capture, Process, Draw, Game Logic, Display
# -----------------------------------------------------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("Unable to read from webcam. Exiting...")
        break

    frame = cv2.flip(frame, 1)             # mirror image
    output = frame.copy()

    now = time.time()
    elapsed = now - game_start_time

    # Increase square speed every SPEED_INCREMENT_INTERVAL seconds (cap at MAX)
    increments = int(elapsed // SPEED_INCREMENT_INTERVAL)
    square_speed = min(SQUARE_BASE_SPEED + increments * SPEED_INCREMENT, MAX_SQUARE_SPEED)

    # Spawn squares at intervals
    if now - last_square_time >= NEW_SQUARE_INTERVAL:
        spawn_square()
        last_square_time = now

    # Spawn circles at intervals
    if now - last_circle_time >= NEW_CIRCLE_INTERVAL:
        spawn_circle()
        last_circle_time = now

    # Update and draw falling objects
    update_falling_objects()
    draw_falling_objects(output)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # Collect all fingertip positions to check collisions
    all_fingertips = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine handedness label
            label = results.multi_handedness[idx].classification[0].label
            color = RIGHT_HAND_COLOR if label == "Right" else LEFT_HAND_COLOR

            # Draw skeleton
            mp_drawing.draw_landmarks(
                image=output,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=color, thickness=LINE_THICKNESS, circle_radius=LANDMARK_RADIUS
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=color, thickness=LINE_THICKNESS, circle_radius=0
                )
            )

            # Finger count & display near wrist
            finger_count, is_right = count_extended_fingers(hand_landmarks)
            wrist = hand_landmarks.landmark[0]
            x_px = int(wrist.x * FRAME_WIDTH)
            y_px = int(wrist.y * FRAME_HEIGHT)
            text = f"{label}: {finger_count}"
            cv2.putText(output, text, (x_px - 40, y_px - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, FINGER_COUNT_COLOR, 2)

            # Fingertip highlights + collect positions
            tips = fingertip_positions(hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            for (fx, fy) in tips:
                all_fingertips.append((fx, fy))
                cv2.circle(output, (fx, fy), FINGERTIP_RADIUS,
                           FINGERTIP_COLOR, -1)

    # Check collisions (squares → +1, circles → –1)
    if all_fingertips:
        check_collisions(all_fingertips)

    # Display score top-left
    cv2.putText(output, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, SCORE_COLOR, 2)

    # Display
    cv2.imshow("Hand Game: Catch Squares, Avoid Circles", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# -----------------------------------------------------------------------------
# 5) Cleanup
# -----------------------------------------------------------------------------
hands.close()
cap.release()
cv2.destroyAllWindows()
