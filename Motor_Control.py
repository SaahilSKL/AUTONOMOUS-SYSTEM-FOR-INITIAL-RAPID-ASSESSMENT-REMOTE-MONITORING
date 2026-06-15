import sys
import time
import threading
from collections import namedtuple

try:
    import RPi.GPIO as GPIO
except ImportError:
    sys.exit("[ERROR] pip install RPi.GPIO")

try:
    import zmq
except ImportError:
    sys.exit("[ERROR] pip install pyzmq")

try:
    import imagezmq
except ImportError:
    sys.exit("[ERROR] pip install imagezmq")

try:
    import cv2
except ImportError:
    sys.exit("[ERROR] pip install opencv-python")


# ═══════════════════════════════════════════════════════════════
# 2.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# GPIO pin numbers (BCM mode)
PIN_ENA = 17;  PIN_IN1 = 6;  PIN_IN2 = 12
PIN_ENB = 27;  PIN_IN3 = 20;  PIN_IN4 = 21

# Direction pins grouped for bulk GPIO.output() calls (one syscall → 4 pins)
_DIR_PINS = (PIN_IN1, PIN_IN2, PIN_IN3, PIN_IN4)

PWM_FREQ   = 1000   # Hz
DRIVE_DUTY = 95 # % — "pace 3 of 5"
TURN_DUTY  = 100     # slightly softer for pivot turns

# Network
PC_IP    = "PC IP"   # ← SET THIS to your PC's local IP
CMD_PORT = 5556              # ZMQ PUB on PC → SUB here
CAM_PORT = 5555              # imagezmq ImageHub on PC

# Camera
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 85
CAM_FPS_CAP  = 25            # max frames/sec sent to PC

# Safety
WATCHDOG_TIMEOUT = 1.5       # seconds — auto-stop if PC goes silent


# ═══════════════════════════════════════════════════════════════
# 3.  GPIO + PWM SETUP
# ═══════════════════════════════════════════════════════════════

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in _DIR_PINS:
    GPIO.setup(pin, GPIO.OUT)
GPIO.output(list(_DIR_PINS), [GPIO.LOW] * 4)   # bulk-zero all direction pins

GPIO.setup(PIN_ENA, GPIO.OUT)
GPIO.setup(PIN_ENB, GPIO.OUT)

pwm_left  = GPIO.PWM(PIN_ENA, PWM_FREQ)
pwm_right = GPIO.PWM(PIN_ENB, PWM_FREQ)
pwm_left.start(0)
pwm_right.start(0)

print("[OK]   GPIO and PWM initialised")


# ═══════════════════════════════════════════════════════════════
# 4.  MOTOR STATE TABLE
#
#  OPTIMISATION A — namedtuple lookup table
#  ─────────────────────────────────────────
#  Old design: each command called 2 helper functions (motor_forward,
#  _set_left, _set_right) full of if/elif chains — 6+ comparisons and
#  4 separate GPIO.output() calls per command on every ZMQ message.
#
#  New design: every possible motor state is pre-built once at startup
#  as a MotorState namedtuple. apply_state() does a single dict lookup
#  (O(1)) and then writes directly. No branching in the hot path.
#
#  OPTIMISATION B — state-diff guard (identity check with `is`)
#  ─────────────────────────────────────────────────────────────
#  Since the table objects are singletons (created once), comparing
#  with `is` is a single pointer comparison — cheaper than == on a
#  namedtuple. If nothing changed we skip ALL GPIO writes entirely.
#  This matters because GPIO.output() and ChangeDutyCycle() involve
#  kernel calls.
#
#  OPTIMISATION C — bulk GPIO.output([pins], [values])
#  ────────────────────────────────────────────────────
#  Writing 4 direction pins with one GPIO.output() call is a single
#  ioctl into the kernel vs 4 separate calls in the old code.
# ═══════════════════════════════════════════════════════════════

MotorState = namedtuple("MotorState", ["in1", "in2", "in3", "in4", "duty_l", "duty_r"])

#                               IN1  IN2  IN3  IN4   L-duty      R-duty
_ST_FORWARD    = MotorState(    1,   0,   1,   0,  DRIVE_DUTY, DRIVE_DUTY)
_ST_REVERSE    = MotorState(    0,   1,   0,   1,  DRIVE_DUTY, DRIVE_DUTY)
_ST_TURN_LEFT  = MotorState(    0,   1,   1,   0,  TURN_DUTY,  TURN_DUTY )  # R fwd, L rev
_ST_TURN_RIGHT = MotorState(    1,   0,   0,   1,  TURN_DUTY,  TURN_DUTY )  # L fwd, R rev
_ST_STOP       = MotorState(    0,   0,   0,   0,  0,          0         )

COMMAND_TABLE = {
    "FORWARD"    : _ST_FORWARD,
    "REVERSE"    : _ST_REVERSE,
    "TURN LEFT"  : _ST_TURN_LEFT,
    "TURN RIGHT" : _ST_TURN_RIGHT,
    "STOP"       : _ST_STOP,
}

_active_state: MotorState = None   # sentinel: nothing applied yet


def apply_state(state: MotorState) -> None:
    """
    Apply a MotorState to GPIO/PWM only when it differs from the
    currently active state. Uses identity check (`is`) on singletons.
    """
    global _active_state
    if state is _active_state:          # OPTIMISATION B — skip redundant writes
        return
    GPIO.output(                        # OPTIMISATION C — single kernel call
        list(_DIR_PINS),
        [state.in1, state.in2, state.in3, state.in4]
    )
    pwm_left.ChangeDutyCycle(state.duty_l)
    pwm_right.ChangeDutyCycle(state.duty_r)
    _active_state = state


def emergency_stop() -> None:
    """Always safe to call — apply_state handles the no-op case."""
    apply_state(_ST_STOP)


print("[OK]   Motor state table ready")
print(f"       Drive : {DRIVE_DUTY}%   Turn : {TURN_DUTY}%\n")


# ═══════════════════════════════════════════════════════════════
# 5.  SHUTDOWN EVENT
#
#  OPTIMISATION D — threading.Event instead of boolean flags
#  ──────────────────────────────────────────────────────────
#  A boolean flag requires a spin-wait loop with time.sleep(). An
#  Event lets blocked threads wake instantly when set(), and
#  Event.wait(timeout) replaces the 0.1 s polling sleep in the old
#  watchdog thread with a proper blocking wait — zero CPU burn while
#  waiting, instant response on shutdown.
# ═══════════════════════════════════════════════════════════════

_stop_event = threading.Event()


# ═══════════════════════════════════════════════════════════════
# 6.  WATCHDOG THREAD
#
#  OPTIMISATION E — time.monotonic() instead of time.time()
#  ─────────────────────────────────────────────────────────
#  time.time() can jump backward or forward if the system clock is
#  adjusted (NTP sync, DST, etc.). time.monotonic() is guaranteed
#  to only move forward — safe for measuring elapsed intervals.
#
#  The old watchdog polled every 0.1 s with time.sleep(0.1).
#  _stop_event.wait(WATCHDOG_TIMEOUT) blocks efficiently until either
#  the timeout fires OR shutdown is requested — zero CPU spin.
# ═══════════════════════════════════════════════════════════════

_last_cmd_mono: float = time.monotonic()


def _watchdog() -> None:
    """
    Fires emergency_stop() if the PC has been silent for
    WATCHDOG_TIMEOUT seconds. Exits cleanly when _stop_event is set.
    """
    while not _stop_event.wait(timeout=WATCHDOG_TIMEOUT):   # OPTIMISATION D+E
        elapsed = time.monotonic() - _last_cmd_mono
        if elapsed >= WATCHDOG_TIMEOUT:
            print(f"[WATCHDOG] No command for {elapsed:.1f}s — stopping motors")
            emergency_stop()
    print("[WATCHDOG] Exited")


threading.Thread(target=_watchdog, daemon=True, name="watchdog").start()
print(f"[OK]   Watchdog armed — auto-stop after {WATCHDOG_TIMEOUT}s silence")


# ═══════════════════════════════════════════════════════════════
# 7.  CAMERA STREAM THREAD
#
#  OPTIMISATION F — FPS cap via _stop_event.wait(frame_interval)
#  ──────────────────────────────────────────────────────────────
#  The old loop ran at full CPU speed, potentially pushing 60-120
#  frames/sec — flooding the ZMQ socket and burning a full CPU core.
#  Capping at CAM_FPS_CAP (25 fps) keeps latency low while leaving
#  headroom for the main motor-control thread.
#
#  _stop_event.wait(remaining) replaces time.sleep() so the thread
#  also wakes instantly on shutdown rather than sleeping through a
#  full frame interval.
#
#  _ENCODE_PARAMS built once at module level — avoids re-creating
#  the list object on every frame (minor, but measurable at 25 fps).
# ═══════════════════════════════════════════════════════════════

_FRAME_INTERVAL = 1.0 / CAM_FPS_CAP
_ENCODE_PARAMS  = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]  # pre-built once


def _camera_stream() -> None:
    sender = imagezmq.ImageSender(connect_to=f"tcp://{PC_IP}:{CAM_PORT}")
    print(f"[CAM]  Sender connected → tcp://{PC_IP}:{CAM_PORT}")

    # Try CSI ribbon cam (Picamera2), fall back to USB cam
    cam = cap = None
    use_picam = False
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        ))
        cam.start()
        use_picam = True
        print("[CAM]  Picamera2 (CSI)")
    except Exception as e:
        print(f"[CAM]  Picamera2 unavailable ({e}), trying USB cam…")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            print("[CAM]  No camera found — stream disabled")
            sender.close()
            return
        print("[CAM]  OpenCV USB camera")

    try:
        while not _stop_event.is_set():
            t0 = time.monotonic()

            try:
                if use_picam:
                    frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        _stop_event.wait(0.05)
                        continue

                ok, jpg_buf = cv2.imencode('.jpg', frame, _ENCODE_PARAMS)
                if ok:
                    sender.send_jpg("raven_rover", jpg_buf.tobytes())

            except Exception as e:
                print(f"[CAM]  Frame error: {e}")

            # OPTIMISATION F — precise FPS cap; wakes instantly on shutdown
            remaining = _FRAME_INTERVAL - (time.monotonic() - t0)
            if remaining > 0:
                _stop_event.wait(remaining)

    finally:
        if use_picam and cam:
            cam.stop()
        elif cap:
            cap.release()
        sender.close()
        print("[CAM]  Stream stopped")


threading.Thread(target=_camera_stream, daemon=True, name="camera").start()


# ═══════════════════════════════════════════════════════════════
# 8.  ZMQ COMMAND SUBSCRIBER
#
#  OPTIMISATION G — zmq.CONFLATE = 1
#  ───────────────────────────────────
#  Without CONFLATE, if the PC sends commands faster than the RPi
#  processes them, stale commands queue up in ZMQ's internal buffer.
#  The rover then executes outdated directions — dangerous near
#  obstacles. CONFLATE keeps only the single latest message, so the
#  rover always acts on the most recent instruction from the PC.
#
#  OPTIMISATION H — ZMQ Poller instead of RCVTIMEO exception path
#  ───────────────────────────────────────────────────────────────
#  RCVTIMEO raises zmq.Again on every timeout — Python exceptions
#  carry significant overhead (traceback construction, etc.). A
#  Poller.poll(timeout_ms) returns an empty dict on timeout with
#  zero exception cost, which is the common case when the rover
#  is executing a sustained command like FORWARD for 2+ seconds.
# ═══════════════════════════════════════════════════════════════

_ctx = zmq.Context()
_sub = _ctx.socket(zmq.SUB)
_sub.setsockopt(zmq.CONFLATE,  1)           # OPTIMISATION G — latest cmd only
_sub.setsockopt(zmq.LINGER,    0)           # don't block on context.term()
_sub.setsockopt_string(zmq.SUBSCRIBE, "")
_sub.connect(f"tcp://{PC_IP}:{CMD_PORT}")

_poller = zmq.Poller()                      # OPTIMISATION H — no exception path
_poller.register(_sub, zmq.POLLIN)

print(f"[OK]   ZMQ SUB connected → tcp://{PC_IP}:{CMD_PORT}")


# ═══════════════════════════════════════════════════════════════
# 9.  MAIN COMMAND LOOP
# ═══════════════════════════════════════════════════════════════

print("\n[OK]   PROJECT RAVEN ROVER — online, awaiting commands…\n")

apply_state(_ST_STOP)    # ensure clean starting state
_current_cmd = "STOP"

try:
    while not _stop_event.is_set():

        # Poll 150 ms — no exception overhead on timeout (OPTIMISATION H)
        ready = dict(_poller.poll(150))

        if _sub not in ready:
            continue    # timeout: watchdog handles prolonged silence

        try:
            msg = _sub.recv_string(zmq.NOBLOCK).strip().upper()
        except zmq.Again:
            continue

        state = COMMAND_TABLE.get(msg)  # O(1) dict lookup (OPTIMISATION A)

        if state is None:
            print(f"[WARN] Unknown command: '{msg}'")
            continue

        # Log only on state transition — not on every repeated command
        if msg != _current_cmd:
            print(f"[CMD]  {_current_cmd:12s} → {msg}")
            _current_cmd = msg

        apply_state(state)              # state-diff + bulk GPIO (OPTIMISATION B+C)
        _last_cmd_mono = time.monotonic()  # OPTIMISATION E — monotonic clock

except KeyboardInterrupt:
    print("\n[SHUTDOWN] KeyboardInterrupt")

finally:
    # ═══════════════════════════════════════════════════════════
    # 10. CLEANUP
    # ═══════════════════════════════════════════════════════════
    print("[SHUTDOWN] Cleaning up…")

    _stop_event.set()       # wake camera + watchdog threads immediately

    emergency_stop()        # park the rover (state-diff-safe, no-op if already stopped)
    time.sleep(0.15)        # let motors coast to a stop before cutting PWM

    pwm_left.stop()
    pwm_right.stop()
    GPIO.cleanup()

    _sub.close()
    _ctx.term()

    print("[SHUTDOWN] Project Raven rover offline.")