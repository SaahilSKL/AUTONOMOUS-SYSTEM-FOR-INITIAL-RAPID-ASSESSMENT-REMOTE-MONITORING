import cv2
import torch
import numpy as np
import imagezmq
import zmq
import sys
import time
import os
from collections import deque
from ultralytics import YOLO

try:
    import keyboard
except ImportError:
    print("[ERROR] 'keyboard' library not found.  Run:  pip install keyboard")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# 1.  CUDA
# ═══════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[OK]   GPU : {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[WARN] CPU mode — performance limited")

# ═══════════════════════════════════════════════════════════════
# 2.  MODELS
# ═══════════════════════════════════════════════════════════════
print("[LOAD] YOLO human detection...")
yolo_model = YOLO('yolov8n.pt')
yolo_model.to(device)

print("[LOAD] Fire detection model...")
fire_model_path = os.path.join("models", "best.pt")
try:
    fire_model = YOLO(fire_model_path)
    fire_model.to(device)
    print(f"[OK]   Fire model loaded")
except Exception as e:
    print(f"[WARN] Fire model failed to load: {e}. Fire detection disabled.")
    fire_model = None  # ← safe sentinel, no fallback to human model

print("[LOAD] MiDaS depth model...")
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
except Exception as e:
    print(f"[ERROR] MiDaS: {e}")
    sys.exit(1)

midas.to(device)
midas.eval()
if device.type == "cuda":
    midas.half()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ═══════════════════════════════════════════════════════════════
# 3.  CONFIG
# ═══════════════════════════════════════════════════════════════
LOWER_FIRE_RED1   = np.array([0,   150, 150])
UPPER_FIRE_RED1   = np.array([10,  255, 255])
LOWER_FIRE_RED2   = np.array([170, 150, 150])
UPPER_FIRE_RED2   = np.array([180, 255, 255])
LOWER_FIRE_ORANGE = np.array([18,  150, 150])
UPPER_FIRE_ORANGE = np.array([35,  255, 255])

MIN_FIRE_AREA         = 1500
OBSTACLE_THRESHOLD    = 0.65
HUMAN_CLOSE_THRESHOLD = 0.20
SMOOTH_WINDOW         = 5

# ═══════════════════════════════════════════════════════════════
# 4.  NETWORK & POLLER
# ═══════════════════════════════════════════════════════════════
image_hub  = imagezmq.ImageHub(open_port='tcp://*:5555')
context    = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:5556")

# Setup ZMQ Poller to prevent blocking recv_jpg()
poller = zmq.Poller()
poller.register(image_hub.zmq_socket, zmq.POLLIN)

print("[OK]   ZMQ sockets and poller bound\n")

# ═══════════════════════════════════════════════════════════════
# 5.  DASHBOARD GEOMETRY
# ═══════════════════════════════════════════════════════════════
DW, DH = 1600, 600          # ← width increased from 1280 to 1600
TOP_H   = 40
BTM_H   = 80
MID_H   = DH - TOP_H - BTM_H    # 480

CAM_W   = 640
DEPTH_W = 640               # ← depth width increased from 320 to 640
MET_W   = DW - CAM_W - DEPTH_W  # 320  (unchanged)

# Panel origins  (x, y)
CAM_X,   CAM_Y   = 0,                  TOP_H
DEPTH_X, DEPTH_Y = CAM_W,              TOP_H
MET_X,   MET_Y   = CAM_W + DEPTH_W,    TOP_H
BTM_Y            = TOP_H + MID_H

# ═══════════════════════════════════════════════════════════════
# 6.  COLOUR PALETTE  (BGR)
# ═══════════════════════════════════════════════════════════════
BG      = ( 20,  20,  26)
PANEL   = ( 32,  32,  42)
PANEL2  = ( 45,  45,  58)
ORANGE  = ( 20, 120, 255)
GREEN   = ( 55, 200,  55)
RED     = ( 50,  50, 215)
YELLOW  = (  0, 210, 240)
CYAN    = (190, 170,  20)
WHITE   = (220, 220, 228)
DIM     = (100, 100, 118)
BORDER  = ( 52,  52,  68)
INDIGO  = (180,  80, 100)

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX

# ═══════════════════════════════════════════════════════════════
# 7.  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════
# Removed redundant Lock as boolean assignments are atomic in Python via GIL
manual_mode   = True      # ← DEFAULT: MANUAL
shutdown_flag = False

def set_manual():
    global manual_mode
    manual_mode = True
    print("[MODE] → MANUAL")

def set_auto():
    global manual_mode
    manual_mode = False
    print("[MODE] → AUTO")

def trigger_shutdown():
    global shutdown_flag
    shutdown_flag = True
    print("[SHUTDOWN] Ctrl+Shift+Q received")

# Register global hotkeys (fire even if OpenCV window isn't focused)
keyboard.add_hotkey('ctrl+shift+m', set_manual,       suppress=True)
keyboard.add_hotkey('ctrl+shift+a', set_auto,         suppress=True)
keyboard.add_hotkey('ctrl+shift+q', trigger_shutdown, suppress=True)

print("[OK]   Hotkeys registered")
print("       Ctrl+Shift+M  →  Manual mode")
print("       Ctrl+Shift+A  →  Auto   mode")
print("       Ctrl+Shift+Q  →  Shutdown\n")

# ═══════════════════════════════════════════════════════════════
# 8.  RUNTIME METRICS
# ═══════════════════════════════════════════════════════════════
decision_history = deque(maxlen=SMOOTH_WINDOW)
cmd_history      = deque(maxlen=7)
frame_times      = deque(maxlen=20)
last_frame_t     = time.time()

# ═══════════════════════════════════════════════════════════════
# 9.  PUSH-BUTTON ARROW READER
# ═══════════════════════════════════════════════════════════════
def read_manual_command() -> str:
    """
    Returns the highest-priority direction key currently held.
    Returns 'STOP' if nothing is pressed.
    Priority: FORWARD > REVERSE > LEFT > RIGHT
    """
    if keyboard.is_pressed('up')   or keyboard.is_pressed('w'):
        return "FORWARD"
    if keyboard.is_pressed('down') or keyboard.is_pressed('s'):
        return "REVERSE"
    if keyboard.is_pressed('left') or keyboard.is_pressed('a'):
        return "TURN LEFT"
    if keyboard.is_pressed('right')or keyboard.is_pressed('d'):
        return "TURN RIGHT"
    return "STOP"

# ═══════════════════════════════════════════════════════════════
# 10. DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════
def rect(c, x, y, w, h, color, fill=True, t=1):
    cv2.rectangle(c, (x, y), (x+w-1, y+h-1), color, -1 if fill else t)

def txt(c, text, x, y, color=WHITE, scale=0.5, thick=1, font=FONT):
    cv2.putText(c, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

def htxt(c, text, cx, y, color=WHITE, scale=0.5, thick=1, font=FONT):
    (w, _), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(c, text, (cx - w//2, y), font, scale, color, thick, cv2.LINE_AA)

def hbar(c, x, y, w, h, val, max_val, fill_col, bg=PANEL2):
    rect(c, x, y, w, h, bg)
    fw = int(w * min(val / max(max_val, 1e-9), 1.0))
    if fw > 0:
        rect(c, x, y, fw, h, fill_col)
    rect(c, x, y, w, h, BORDER, fill=False)

def sig_bars(c, x, y, fps):
    bars = 4 if fps>22 else 3 if fps>15 else 2 if fps>8 else 1 if fps>3 else 0
    for i in range(4):
        bh = 6 + i * 5
        bx = x + i * 9
        by = y - bh
        col = GREEN if i < bars else PANEL2
        rect(c, bx, by, 6, bh, col)
    return bars

def divider(c, x, y, w):
    cv2.line(c, (x, y), (x + w, y), BORDER, 1)

def smooth(new_cmd: str) -> str:
    if new_cmd in ("STOP", "REVERSE"):
        decision_history.clear()
        decision_history.append(new_cmd)
        return new_cmd
    decision_history.append(new_cmd)
    return max(set(decision_history), key=decision_history.count)

# ═══════════════════════════════════════════════════════════════
# 11. PANEL RENDERERS
# ═══════════════════════════════════════════════════════════════
def draw_top_bar(canvas, mode, status, fps):
    rect(canvas, 0, 0, DW, TOP_H, (16, 16, 22))

    # Logo block
    rect(canvas, 8, 7, 26, 26, ORANGE)
    txt(canvas, "R", 13, 28, (16,16,22), 0.72, 2)
    txt(canvas, "PROJECT RAVEN", 42, 26, WHITE, 0.58, 1)

    # Mode pill
    pc  = (25, 110, 25) if mode == "AUTO" else (100, 45, 160)
    lbl = "● AUTO  [Ctrl+Shift+M for Manual]" if mode=="AUTO" else "● MANUAL  [Ctrl+Shift+A for Auto]"
    rect(canvas, 208, 9, 310, 22, pc)
    txt(canvas, lbl, 216, 25, WHITE, 0.38, 1)

    # Status
    txt(canvas, status, 530, 26, DIM, 0.42)

    # FPS
    fc = GREEN if fps>15 else YELLOW if fps>8 else RED
    txt(canvas, f"FPS {fps:4.1f}", DW-230, 26, fc, 0.46)

    # Shutdown hint
    txt(canvas, "Ctrl+Shift+Q: Quit", DW-155, 26, DIM, 0.36)

    cv2.line(canvas, (0, TOP_H), (DW, TOP_H), BORDER, 1)

def draw_camera_panel(canvas, frame, roi_top, roi_bot, third_w):
    if frame is not None:
        canvas[CAM_Y:CAM_Y+MID_H, CAM_X:CAM_X+CAM_W] = cv2.resize(frame, (CAM_W, MID_H))
    else:
        rect(canvas, CAM_X, CAM_Y, CAM_W, MID_H, PANEL)
        htxt(canvas, "NO SIGNAL", CAM_X + CAM_W//2, CAM_Y + MID_H//2, DIM, 0.8)

    # ROI overlay
    ry1 = CAM_Y + roi_top
    ry2 = CAM_Y + roi_bot
    cv2.rectangle(canvas, (CAM_X, ry1), (CAM_X+CAM_W, ry2), (160,160,160), 1)
    cv2.line(canvas, (CAM_X+third_w, ry1),   (CAM_X+third_w, ry2),   (110,110,110), 1)
    cv2.line(canvas, (CAM_X+2*third_w, ry1), (CAM_X+2*third_w, ry2), (110,110,110), 1)

    # Corner brackets (tactical)
    L, T = 22, 2
    for (px, py, sx, sy) in [
        (CAM_X,        CAM_Y,         1,  1),
        (CAM_X+CAM_W,  CAM_Y,        -1,  1),
        (CAM_X,        CAM_Y+MID_H,   1, -1),
        (CAM_X+CAM_W,  CAM_Y+MID_H,  -1, -1),
    ]:
        cv2.line(canvas, (px, py), (px+L*sx, py),     ORANGE, T)
        cv2.line(canvas, (px, py), (px, py+L*sy),     ORANGE, T)

    rect(canvas, CAM_X, CAM_Y, CAM_W, MID_H, BORDER, fill=False)

def draw_depth_panel(canvas, depth_bgr, roi_top, roi_bot):
    rect(canvas, DEPTH_X, DEPTH_Y, DEPTH_W, MID_H, PANEL)

    # Header
    rect(canvas, DEPTH_X, DEPTH_Y, DEPTH_W, 24, PANEL2)
    txt(canvas, "DEPTH MAP", DEPTH_X+10, DEPTH_Y+17, DIM, 0.44)
    txt(canvas, "MiDaS_small", DEPTH_X+DEPTH_W-90, DEPTH_Y+17, ORANGE, 0.38)
    cv2.line(canvas, (DEPTH_X, DEPTH_Y+24), (DEPTH_X+DEPTH_W, DEPTH_Y+24), BORDER, 1)

    if depth_bgr is not None:
        dh   = MID_H - 24
        dd   = cv2.resize(depth_bgr, (DEPTH_W, dh))
        # ROI lines on depth map
        sy   = dh / 480
        rt   = int(roi_top * sy)
        rb   = int(roi_bot * sy)
        tw   = DEPTH_W // 3
        cv2.rectangle(dd, (0, rt), (DEPTH_W, rb), (200,200,200), 1)
        cv2.line(dd, (tw, rt),    (tw, rb),    (160,160,160), 1)
        cv2.line(dd, (2*tw, rt),  (2*tw, rb),  (160,160,160), 1)
        canvas[DEPTH_Y+24 : DEPTH_Y+MID_H, DEPTH_X : DEPTH_X+DEPTH_W] = dd
    else:
        htxt(canvas, "NO SIGNAL", DEPTH_X + DEPTH_W//2, DEPTH_Y + MID_H//2, DIM, 0.6)

    rect(canvas, DEPTH_X, DEPTH_Y, DEPTH_W, MID_H, BORDER, fill=False)

def draw_metrics_panel(canvas, people, fire_ext, fps, path_p,
                       cmd_hist, mode, final_cmd, dec_color):
    rect(canvas, MET_X, MET_Y, MET_W, MID_H, PANEL)

    # Header
    rect(canvas, MET_X, MET_Y, MET_W, 28, PANEL2)
    txt(canvas, "SYSTEM METRICS", MET_X+10, MET_Y+20, WHITE, 0.48)
    cv2.line(canvas, (MET_X, MET_Y+28), (MET_X+MET_W, MET_Y+28), BORDER, 1)

    pad = 12
    oy  = MET_Y + 42

    # ── Active Command ──────────────────────────────────────────
    rect(canvas, MET_X+pad, oy-4, MET_W-2*pad, 36, PANEL2)
    htxt(canvas, final_cmd, MET_X + MET_W//2, oy+22, dec_color, 0.72, 2)
    mode_tag = "MANUAL" if mode=="MANUAL" else "AUTO"
    htxt(canvas, f"[ {mode_tag} ]", MET_X + MET_W//2, oy+38, DIM, 0.35)
    oy += 56
    divider(canvas, MET_X+pad, oy, MET_W-2*pad)
    oy += 12

    # ── People ─────────────────────────────────────────────────
    txt(canvas, "PEOPLE DETECTED", MET_X+pad, oy, DIM, 0.38)
    oy += 18
    pc = RED if people > 0 else GREEN
    cv2.putText(canvas, str(people), (MET_X+pad, oy+28),
                FONT, 1.6, pc, 2, cv2.LINE_AA)
    txt(canvas, "in frame", MET_X+pad+40, oy+14, DIM, 0.35)
    oy += 48
    divider(canvas, MET_X+pad, oy, MET_W-2*pad)
    oy += 12

    # ── Fire Extent ────────────────────────────────────────────
    txt(canvas, "FIRE/SMOKE EXTENT", MET_X+pad, oy, DIM, 0.38)
    oy += 18
    fp  = fire_ext * 100
    fc  = RED if fp>15 else YELLOW if fp>5 else GREEN
    txt(canvas, f"{fp:.1f}%", MET_X+pad, oy+14, fc, 0.72, 2)
    hbar(canvas, MET_X+pad, oy+20, MET_W-2*pad, 9, fp, 100, fc)
    oy += 42
    divider(canvas, MET_X+pad, oy, MET_W-2*pad)
    oy += 12

    # ── Signal Strength ────────────────────────────────────────
    txt(canvas, "SIGNAL STRENGTH", MET_X+pad, oy, DIM, 0.38)
    oy += 22
    bars = sig_bars(canvas, MET_X+pad, oy, fps)
    slbl = ["NO SIGNAL","POOR","FAIR","GOOD","STRONG"][bars]
    scol = [RED, RED, YELLOW, GREEN, GREEN][bars]
    txt(canvas, slbl, MET_X+pad+46, oy, scol, 0.50)
    txt(canvas, f"{fps:.0f} fps", MET_X+MET_W-56, oy, DIM, 0.38)
    oy += 22
    divider(canvas, MET_X+pad, oy, MET_W-2*pad)
    oy += 12

    # ── Path Probability ───────────────────────────────────────
    txt(canvas, "PATH CLEAR PROB", MET_X+pad, oy, DIM, 0.38)
    oy += 18
    pp  = path_p * 100
    ppc = GREEN if pp>60 else YELLOW if pp>35 else RED
    txt(canvas, f"{pp:.0f}%", MET_X+pad, oy+14, ppc, 0.72, 2)
    hbar(canvas, MET_X+pad, oy+20, MET_W-2*pad, 9, pp, 100, ppc)
    oy += 42
    divider(canvas, MET_X+pad, oy, MET_W-2*pad)
    oy += 12

    # ── Command History ────────────────────────────────────────
    txt(canvas, "CMD HISTORY", MET_X+pad, oy, DIM, 0.38)
    oy += 16
    hist = list(cmd_hist)
    for i, cmd in enumerate(reversed(hist[-6:])):
        fade  = max(0.25, 1.0 - i * 0.14)
        col   = tuple(int(v * fade) for v in WHITE)
        mark  = ">" if i == 0 else " "
        txt(canvas, f"{mark} {cmd}", MET_X+pad, oy + i*17, col, 0.36)

    rect(canvas, MET_X, MET_Y, MET_W, MID_H, BORDER, fill=False)

def draw_bottom_bar(canvas, l, c, r, mode):
    rect(canvas, 0, BTM_Y, DW, BTM_H, (16,16,22))
    cv2.line(canvas, (0, BTM_Y), (DW, BTM_Y), BORDER, 1)

    # Lane score bars
    lx, bw = 14, 150
    for label, val, ox in [("LEFT", l, lx),
                            ("CENTER", c, lx+192),
                            ("RIGHT",  r, lx+384)]:
        bc = GREEN if val>0.6 else YELLOW if val>0.3 else RED
        txt(canvas, label, ox, BTM_Y+20, DIM, 0.38)
        hbar(canvas, ox, BTM_Y+26, bw, 10, val*100, 100, bc)
        txt(canvas, f"{val:.2f}", ox+bw+6, BTM_Y+36, bc, 0.36)

    # Hotkey cheat sheet (right side)
    hints = [
        ("Ctrl+Shift+M", "Manual mode"),
        ("Ctrl+Shift+A", "Auto   mode"),
        ("Arrow keys",   "Drive  (push-button)"),
    ]
    hx = 580
    txt(canvas, "HOTKEYS", hx, BTM_Y+20, DIM, 0.36)
    for i, (k, v) in enumerate(hints):
        txt(canvas, k, hx,    BTM_Y+36+i*16, ORANGE, 0.36)
        txt(canvas, v, hx+115,BTM_Y+36+i*16, DIM,    0.36)

    # Vertical divider before command block
    cv2.line(canvas, (DW-320, BTM_Y), (DW-320, DH), BORDER, 1)

    # Command block
    rect(canvas, DW-318, BTM_Y+8, 310, BTM_H-16, PANEL2)
    rect(canvas, DW-318, BTM_Y+8, 310, BTM_H-16, BORDER, fill=False)
    mode_col = INDIGO if mode=="MANUAL" else GREEN
    txt(canvas, mode, DW-306, BTM_Y+26, mode_col, 0.42)
    cv2.line(canvas, (DW-318, BTM_Y+34), (DW-8, BTM_Y+34), BORDER, 1)


# ═══════════════════════════════════════════════════════════════
# 12. MAIN LOOP
# ═══════════════════════════════════════════════════════════════
print("[OK]   SYSTEM ONLINE — default MANUAL mode\n")

try:
    with torch.no_grad():
        while not shutdown_flag:

            is_manual = manual_mode

            # ── Receive frame (Non-Blocking) ───────────────────────
            events = dict(poller.poll(timeout=100)) # 100ms timeout prevents GUI freeze
            frame = None
            
            if image_hub.zmq_socket in events:
                try:
                    rpi_name, jpg_buf = image_hub.recv_jpg()
                    arr   = np.frombuffer(jpg_buf, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    image_hub.send_reply(b'OK')
                except Exception as e:
                    print(f"[WARN] Frame receive error: {e}")

            # FPS calculation
            now = time.time()
            frame_times.append(now - last_frame_t)
            last_frame_t = now
            signal_fps = 1.0 / (float(np.mean(frame_times)) + 1e-9)

            # Per-frame reset
            raw_cmd       = "STOP"
            status        = "SCANNING"
            dec_color     = DIM
            fire_detected = False
            human_close   = False
            m_people      = 0
            m_fire_ext    = 0.0
            depth_bgr     = None
            
            # Default UI parameters if no frame
            roi_top, roi_bot = int(480 * 0.40), int(480 * 0.90)
            third_w = 640 // 3
            left_sc, center_sc, right_sc, path_prob = 0.0, 0.0, 0.0, 0.0

            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                H, W  = frame.shape[:2]

                # ── A. AI Fire Detection ───────────────────────────────
                if fire_model is not None:
                    for result in fire_model(frame, verbose=False):
                        for box in result.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls  = int(box.cls[0])          # 0 = Fire, 1 = Smoke
                            label = "FIRE" if cls == 0 else "SMOKE"
                            if conf > 0.4:
                                fire_detected = True
                                raw_cmd       = "STOP"
                                status        = f"AI {label} DETECTED"
                                dec_color     = RED
                                m_fire_ext    = max(m_fire_ext, (x2-x1)*(y2-y1)/(W*H))
                                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                                cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                                            FONT,0.45,(0,0,255),1,cv2.LINE_AA)

                # ── B. HSV Fire (backup) ───────────────────────────────
                if not fire_detected:
                    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mk1  = cv2.inRange(hsv, LOWER_FIRE_RED1,   UPPER_FIRE_RED1)
                    mk2  = cv2.inRange(hsv, LOWER_FIRE_RED2,   UPPER_FIRE_RED2)
                    mk3  = cv2.inRange(hsv, LOWER_FIRE_ORANGE, UPPER_FIRE_ORANGE)
                    fmask = cv2.bitwise_or(mk1, cv2.bitwise_or(mk2, mk3))
                    fpix  = cv2.countNonZero(fmask)
                    m_fire_ext = fpix / (W * H)
                    if fpix > MIN_FIRE_AREA:
                        fire_detected = True
                        raw_cmd = "STOP"
                        status  = "HSV FIRE DETECTED"
                        dec_color = RED
                        cnts,_ = cv2.findContours(fmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in cnts:
                            fx,fy,fw,fh = cv2.boundingRect(cnt)
                            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,165,255),2)

                # ── C. Depth Estimation ────────────────────────────────
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch = transform(rgb).to(device)
                if device.type == "cuda":
                    batch = batch.half()

                pred = midas(batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(H,W), mode="bicubic", align_corners=False
                ).squeeze()
                raw_d      = pred.cpu().float().numpy()
                depth_norm = cv2.normalize(raw_d, None, 0, 1, cv2.NORM_MINMAX)
                depth_norm = cv2.GaussianBlur(depth_norm, (11,11), 0)

                roi_top = int(H * 0.40)
                roi_bot = int(H * 0.90)
                nav     = depth_norm[roi_top:roi_bot, :]
                safe    = 1.0 - (nav > OBSTACLE_THRESHOLD).astype(np.float32)

                third_w    = W // 3
                left_sc    = float(np.mean(safe[:,         :third_w]))
                center_sc  = float(np.mean(safe[:, third_w  :2*third_w]))
                right_sc   = float(np.mean(safe[:, 2*third_w:]))
                path_prob  = center_sc

                depth_bgr = cv2.applyColorMap((depth_norm*255).astype(np.uint8),
                                              cv2.COLORMAP_MAGMA)

                # ── D. Human Detection ─────────────────────────────────
                if not fire_detected:
                    for result in yolo_model(frame, verbose=False, classes=[0]):
                        for box in result.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            m_people += 1
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,120,30),2)
                            cv2.putText(frame,"HUMAN",(x1,y1-8),
                                        FONT,0.38,(255,120,30),1,cv2.LINE_AA)
                            if (x2-x1)*(y2-y1) > W*H*HUMAN_CLOSE_THRESHOLD:
                                human_close = True
            else:
                status = "NO SIGNAL"

            # ── E. Command Decision ────────────────────────────────
            if is_manual:
                # ── MANUAL: push-button keyboard state ──────────
                final_cmd = read_manual_command()
                # Fire override: safety first — manual can't drive into fire
                if fire_detected:
                    final_cmd = "STOP"
                    status    = "FIRE — MANUAL BLOCKED"
                    dec_color = RED
                else:
                    status = "MANUAL DRIVE" if frame is not None else "MANUAL DRIVE (BLIND)"
                    dec_color = {
                        "FORWARD"   : GREEN,
                        "REVERSE"   : CYAN,
                        "TURN LEFT" : YELLOW,
                        "TURN RIGHT": YELLOW,
                        "STOP"      : DIM,
                    }.get(final_cmd, DIM)

            else:
                # ── AUTO: depth + human AI navigation ───────────
                if frame is None:
                    final_cmd = "STOP"
                    status    = "AUTO BLOCKED (NO SIGNAL)"
                    dec_color = RED
                elif not fire_detected:
                    if human_close:
                        status = "HUMAN AVOIDANCE"
                        if   left_sc > right_sc and left_sc > 0.3:
                            raw_cmd="TURN LEFT";  dec_color=YELLOW
                        elif right_sc > left_sc and right_sc > 0.3:
                            raw_cmd="TURN RIGHT"; dec_color=YELLOW
                        else:
                            raw_cmd="STOP";       dec_color=RED
                    else:
                        status = "DEPTH NAVIGATION"
                        if   center_sc > 0.6:
                            raw_cmd="FORWARD";    dec_color=GREEN
                        elif left_sc > right_sc and left_sc > 0.3:
                            raw_cmd="TURN LEFT";  dec_color=YELLOW
                        elif right_sc > left_sc and right_sc > 0.3:
                            raw_cmd="TURN RIGHT"; dec_color=YELLOW
                        else:
                            raw_cmd="REVERSE";    dec_color=CYAN
                final_cmd = smooth(raw_cmd)

            # ── F. Publish to RPi ──────────────────────────────────
            pub_socket.send_string(final_cmd)
            if not cmd_history or cmd_history[-1] != final_cmd:
                cmd_history.append(final_cmd)

            # Overlay HUD text on camera frame
            if frame is not None:
                fc_col = dec_color
                cv2.putText(frame, f"{final_cmd}", (20,42),
                            FONT, 1.0, fc_col, 3, cv2.LINE_AA)
                cv2.putText(frame, status, (20, 74),
                            FONTS, 0.58, WHITE, 1, cv2.LINE_AA)

            # ── G. Render Dashboard ────────────────────────────────
            canvas = np.full((DH, DW, 3), BG, dtype=np.uint8)

            mode_str = "MANUAL" if is_manual else "AUTO"
            draw_top_bar(canvas, mode_str, status, signal_fps)
            draw_camera_panel(canvas, frame, roi_top, roi_bot, third_w)
            draw_depth_panel(canvas, depth_bgr, roi_top, roi_bot)
            draw_metrics_panel(canvas, m_people, m_fire_ext, signal_fps,
                               path_prob, cmd_history, mode_str,
                               final_cmd, dec_color)
            draw_bottom_bar(canvas, left_sc, center_sc, right_sc, mode_str)

            # ── Large centred command in bottom block ──────────────
            htxt(canvas, final_cmd, DW - 163, BTM_Y + BTM_H//2 + 10,
                 dec_color, 0.78, 2)

            cv2.imshow("PROJECT RAVEN — Command Center", canvas)

            # Keep OpenCV window responsive
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # ═══════════════════════════════════════════════════════════════
    # 13. CLEANUP
    # ═══════════════════════════════════════════════════════════════
    print("\n[SHUTDOWN] Initiating cleanup sequence...")
    pub_socket.send_string("STOP")   # park the rover before exit
    time.sleep(0.1)

    cv2.destroyAllWindows()
    image_hub.close()
    pub_socket.close()
    context.term()
    keyboard.unhook_all()
    print("[SHUTDOWN] Project Raven offline.")