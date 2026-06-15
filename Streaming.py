import socket
import time
import cv2
import imagezmq
import threading
import zmq

# --- CONFIGURATION ---
TARGET_IP = "PC IP"
JPEG_QUALITY = 80
FPS_LIMIT = 15
FRAME_TIME = 1 / FPS_LIMIT   # 0.1 seconds per frame


# --- THREADED CAMERA CLASS ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# --- MAIN SETUP ---
sender = imagezmq.ImageSender(connect_to=f'tcp://{TARGET_IP}:5555')
rpi_name = socket.gethostname()

# Start threaded camera
cam = VideoStream(src=0).start()
time.sleep(2.0)  # camera warmup

print(f"Streaming at {FPS_LIMIT} FPS to {TARGET_IP}...")

try:
    while True:

        start_time = time.time()

        frame = cam.read()

        # JPEG compression
        ret_code, jpg_buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        # send frame
        sender.send_jpg(rpi_name, jpg_buffer)

        # --- FPS CONTROL ---
        elapsed = time.time() - start_time
        sleep_time = FRAME_TIME - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    cam.stop()
    print("Stream stopped.")