import cv2
import math
import random
import time
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

W, H = 640, 480
ui_center = [W // 2, H // 2]

BASE_MIN_R = 90
BASE_MAX_R = 260

NUM_PARTICLES = 70
particles = []
for _ in range(NUM_PARTICLES):
    r = random.uniform(BASE_MIN_R, BASE_MAX_R)
    ang = random.uniform(0, 2 * math.pi)
    spd = random.uniform(0.002, 0.01)
    base_size = random.randint(8, 16)
    particles.append([0, r, ang, spd, base_size, 0.0, 0.0, 1.0])

ui_dragging = False
ui_drag_hand = -1

selected_particle = -1
particle_holding_hand = -1

prev_pinch = [False, False]

smooth_ui_angle = 0.0
ui_scale = 1.0
smooth_ui_scale = 1.0

trail = np.zeros((H, W, 3), dtype=np.uint8)

MAIN = (255, 220, 80)
BRIGHT = (255, 255, 150)
DIM = (150, 140, 50)
WHITE = (255, 255, 255)

MAGNET_RADIUS = 400
SNAP_STRENGTH = 0.75

UI_GRAB_MIN = 50
UI_GRAB_MAX = 320

def clamp(v, a, b):
    return max(a, min(b, v))

def distance_px(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def dist_to_center(x, y):
    return math.hypot(x - ui_center[0], y - ui_center[1])

def polar_to_xy(r, ang, scale, rot_deg):
    cx, cy = ui_center
    rot = math.radians(rot_deg)
    a = ang + rot
    x = cx + (r * scale) * math.cos(a)
    y = cy + (r * scale) * math.sin(a)
    return x, y

def glow_dot_fast(img, x, y, r, color):
    x, y, r = int(x), int(y), int(r)
    overlay = img.copy()
    cv2.circle(overlay, (x, y), r + 8, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)
    cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)

def draw_jarvis_ui(layer, t, scale):
    cx, cy = ui_center
    rot1 = (t * 90) % 360
    rot2 = (360 - (t * 130) % 360)

    def S(v):
        return int(v * scale)

    cv2.circle(layer, (cx, cy), S(110), DIM, 1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), S(170), MAIN, 1, cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), S(250), DIM, 1, cv2.LINE_AA)

    cv2.ellipse(layer, (cx, cy), (S(140), S(140)), 0, rot1, rot1 + 90, BRIGHT, 2, cv2.LINE_AA)
    cv2.ellipse(layer, (cx, cy), (S(210), S(210)), 0, rot2, rot2 + 70, MAIN, 2, cv2.LINE_AA)

    cv2.line(layer, (cx - S(280), cy), (cx + S(280), cy), DIM, 1, cv2.LINE_AA)
    cv2.line(layer, (cx, cy - S(280)), (cx, cy + S(280)), DIM, 1, cv2.LINE_AA)

def rotate_layer(layer, angle_deg, center):
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

def hand_rotation_angle_deg(hand_landmarks):
    wx = hand_landmarks.landmark[0].x * W
    wy = hand_landmarks.landmark[0].y * H
    ix = hand_landmarks.landmark[5].x * W
    iy = hand_landmarks.landmark[5].y * H
    return math.degrees(math.atan2(iy - wy, ix - wx))

def find_best_particle(cursor_x, cursor_y, scale, rot_deg, max_dist=MAGNET_RADIUS):
    best = -1
    best_d = 999999

    for i, p in enumerate(particles):
        mode, r, ang, spd, base_size, fx, fy, extra_scale = p

        if mode == 0:
            px, py = polar_to_xy(r, ang, scale, rot_deg)
        else:
            px, py = fx, fy

        d = math.hypot(px - cursor_x, py - cursor_y)
        if d < best_d and d < max_dist:
            best_d = d
            best = i

    return best

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

start = time.time()
frame_count = 0
cached_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (W, H))

    trail = (trail * 0.87).astype(np.uint8)

    for i, p in enumerate(particles):
        if i == selected_particle:
            continue
        if p[0] == 0:
            p[2] += p[3]

    frame_count += 1
    if frame_count % 2 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cached_result = hands.process(rgb)

    cursors = [None, None]
    pinch_states = [False, False]
    hand_centers = []
    target_angle = smooth_ui_angle

    if cached_result and cached_result.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(cached_result.multi_hand_landmarks[:2]):
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            wx = int(hand_landmarks.landmark[0].x * W)
            wy = int(hand_landmarks.landmark[0].y * H)
            hand_centers.append((wx, wy))

            ix = int(hand_landmarks.landmark[8].x * W)
            iy = int(hand_landmarks.landmark[8].y * H)
            tx = int(hand_landmarks.landmark[4].x * W)
            ty = int(hand_landmarks.landmark[4].y * H)

            cursors[hand_index] = (ix, iy)

            pinch_dist = distance_px(ix, iy, tx, ty)
            pinch = pinch_dist < 35
            pinch_states[hand_index] = pinch

            if hand_index == 0:
                target_angle = hand_rotation_angle_deg(hand_landmarks)

            pinch_started = pinch and not prev_pinch[hand_index]

            if pinch_started and selected_particle == -1 and not ui_dragging:
                candidate = find_best_particle(ix, iy, smooth_ui_scale, smooth_ui_angle)

                if candidate != -1:
                    selected_particle = candidate
                    particle_holding_hand = hand_index

                    p = particles[selected_particle]
                    if p[0] == 0:
                        px, py = polar_to_xy(p[1], p[2], smooth_ui_scale, smooth_ui_angle)
                        p[5], p[6] = px, py
                        p[0] = 1

                    p[7] = 1.0

                else:
                    dcenter = dist_to_center(ix, iy)
                    if UI_GRAB_MIN * smooth_ui_scale <= dcenter <= UI_GRAB_MAX * smooth_ui_scale:
                        ui_dragging = True
                        ui_drag_hand = hand_index

            if ui_dragging and ui_drag_hand == hand_index and selected_particle == -1:
                if pinch:
                    ui_center[0] = ix
                    ui_center[1] = iy
                else:
                    ui_dragging = False
                    ui_drag_hand = -1

            if selected_particle != -1 and particle_holding_hand == hand_index:
                if pinch:
                    p = particles[selected_particle]
                    p[5] = p[5] * (1 - SNAP_STRENGTH) + ix * SNAP_STRENGTH
                    p[6] = p[6] * (1 - SNAP_STRENGTH) + iy * SNAP_STRENGTH
                else:
                    selected_particle = -1
                    particle_holding_hand = -1

            prev_pinch[hand_index] = pinch

    if len(hand_centers) == 2:
        d = distance_px(hand_centers[0][0], hand_centers[0][1], hand_centers[1][0], hand_centers[1][1])
        ui_scale_val = (d - 80) / (320 - 80)
        ui_scale_val = clamp(ui_scale_val * 1.2, 0.7, 1.8)
        ui_scale = ui_scale_val

    smooth_ui_scale = smooth_ui_scale * 0.90 + ui_scale * 0.10
    smooth_ui_angle = smooth_ui_angle * 0.90 + target_angle * 0.10

    t = time.time() - start

    out = cv2.addWeighted(frame, 1.0, trail, 0.75, 0)

    ui_layer = np.zeros((H, W, 3), dtype=np.uint8)
    draw_jarvis_ui(ui_layer, t, smooth_ui_scale)
    rotated_ui = rotate_layer(ui_layer, smooth_ui_angle, tuple(ui_center))
    out = cv2.addWeighted(out, 1.0, rotated_ui, 1.0, 0)

    for i, p in enumerate(particles):
        mode, r, ang, spd, base_size, fx, fy, extra_scale = p

        if mode == 0:
            x, y = polar_to_xy(r, ang, smooth_ui_scale, smooth_ui_angle)
        else:
            x, y = fx, fy

        size = int(base_size * smooth_ui_scale * 1.6)
        size = max(2, size)

        glow_dot_fast(trail, x, y, 2, DIM)

        if i == selected_particle:
            glow_dot_fast(out, x, y, size + 5, WHITE)
        else:
            glow_dot_fast(out, x, y, size, BRIGHT)

    for i, cur in enumerate(cursors):
        if cur is None:
            continue
        x, y = cur
        cv2.circle(out, (x, y), 10, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(out, "Pinch Particle = Move Particle | Pinch Empty = Move UI | Two Hands = Zoom UI",
                (6, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 2, cv2.LINE_AA)

    cv2.imshow("Jarvis - Perfect UI Move + Particle Select", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
