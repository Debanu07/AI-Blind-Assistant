import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time
# TTS control
_tts_lock = threading.Semaphore(1)
last_spoken = {}
COOLDOWN = 4

def speak(text, label_key):

    def _speak():
        if _tts_lock.acquire(blocking=False):
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            finally:
                _tts_lock.release()

    threading.Thread(target=_speak, daemon=True).start()
# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Label mapping
LABEL_MAP = {
    "cell phone": "phone"
}

CONFIDENCE = 0.45
previous_objects = set()

try:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_width = frame.shape[1]

        results = model(frame, conf=CONFIDENCE)

        current_objects = set()
        # Detection loop
        for box in results[0].boxes:

            class_id = int(box.cls[0])
            label = model.names[class_id]

            label = LABEL_MAP.get(label, label)

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = (x1 + x2) / 2

            # direction
            if center_x < frame_width / 3:
                position = "left"
            elif center_x < 2 * frame_width / 3:
                position = "ahead"
            else:
                position = "right"

            # distance
            area = (x2 - x1) * (y2 - y1)

            if area > 150000:
                distance = "very close"
            elif area > 60000:
                distance = "near"
            else:
                distance = "far"

            description = f"{label} {distance} {position}"

            current_objects.add(description)
        # Speak new objects with cooldown
        new_objects = current_objects - previous_objects

        for obj in new_objects:

            key = obj.split()[0]  

            if time.time() - last_spoken.get(key, 0) > COOLDOWN:

                print("Speaking:", obj)

                speak(obj, key)

                last_spoken[key] = time.time()

        previous_objects = current_objects

        # Display overlay
        display_texts = sorted(current_objects)

        annotated_frame = results[0].plot()

        y = 30
        for text in display_texts:

            cv2.putText(
                annotated_frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            y += 30

        cv2.imshow("AI Blind Assistant", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()

    cv2.destroyAllWindows()
