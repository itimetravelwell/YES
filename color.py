import cv2
import numpy as np
import time
import os
import sys

# Suppress OpenCV warnings by redirecting stderr
sys.stderr = open(os.devnull, 'w')

# Set Qt platform to use "xcb" plugin
os.environ["QT_QPA_PLATFORM"] = "xcb"

time.sleep(2)  # wait for 2 seconds

# YOLO model files (default)
yolo_dir = "/home/imm/Desktop/YES/yolo"
weights_path = os.path.join(yolo_dir, "yolov3-tiny.weights")
config_path = os.path.join(yolo_dir, "yolov3-tiny.cfg")
names_path = os.path.join(yolo_dir, "coco.names")

# Load YOLO model
if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(names_path):
    print("Error: YOLO model files not found.")
    exit()

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Disable CUDA and use the CPU backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# URL of the video feed
video_url = "http://172.20.10.6/stream"

# Open the video feed
cap = cv2.VideoCapture(video_url)

# Restore stderr after suppressing warnings
sys.stderr = sys.__stderr__

if not cap.isOpened():
    print("Error: Unable to open video feed.")
    exit()

# Define the codec and create a VideoWriter object for saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
output_file = "person_detection_output.avi"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if frame_width == 0 or frame_height == 0:
    print("Error: Invalid frame dimensions.")
    exit()

out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

# Specify the class to detect (only "person")
target_classes = ["person"]

# Detection smoothing parameters
detection_history = []
history_length = 10  # Number of frames to consider for smoothing

# Function to process video frames
frame_skip_interval = 3  # Process every 3rd frame
frame_count = 0
def process_video():
    global frame_count, detection_history
    frame_count += 1
    if frame_count % frame_skip_interval != 0:
        return True

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame. Check the video feed URL or network connection.")
        return False

    # --- HSV Thresholding for Orange Detection (general orange, not just bright) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])  # General orange
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_orange = cv2.erode(mask_orange, None, iterations=2)
    mask_orange = cv2.dilate(mask_orange, None, iterations=2)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orange_detected = False
    for cnt in contours_orange:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
            cv2.putText(frame, "Orange", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            orange_detected = True

    # --- HSV Thresholding for Lime Green Detection ---
    lower_lime = np.array([40, 100, 100])  # Lower bound for lime green
    upper_lime = np.array([75, 255, 255])  # Upper bound for lime green
    mask_lime = cv2.inRange(hsv, lower_lime, upper_lime)
    mask_lime = cv2.erode(mask_lime, None, iterations=2)
    mask_lime = cv2.dilate(mask_lime, None, iterations=2)
    contours_lime, _ = cv2.findContours(mask_lime, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lime_detected = False
    for cnt in contours_lime:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Lime Green", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            lime_detected = True

    # --- HSV Thresholding for Light Blue Detection ---
    lower_light_blue = np.array([85, 50, 150])   # Lower bound for light blue
    upper_light_blue = np.array([105, 255, 255]) # Upper bound for light blue
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
    mask_light_blue = cv2.erode(mask_light_blue, None, iterations=2)
    mask_light_blue = cv2.dilate(mask_light_blue, None, iterations=2)
    contours_light_blue, _ = cv2.findContours(mask_light_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    light_blue_detected = False
    for cnt in contours_light_blue:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, "Light Blue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            light_blue_detected = True

    # --- YOLO Person Detection (unchanged) ---
    resized_frame = cv2.resize(frame, (320, 320))
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    height, width, _ = frame.shape
    is_person_detected = False
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                class_name = classes[class_id]
                if class_name in target_classes:
                    is_person_detected = True
                    center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (255, 0, 0), 2)
                    cv2.putText(frame, f"{class_name}: {int(confidence * 100)}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    detection_history.append(is_person_detected)
    if len(detection_history) > history_length:
        detection_history.pop(0)
    smoothed_detection = sum(detection_history) > (history_length // 2)

    # Display detection status on the video feed
    status_text = "Person Detected" if smoothed_detection else "No Person Detected"
    status_color = (255, 0, 0) if smoothed_detection else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Display orange detection status
    orange_status_text = "Orange Detected" if orange_detected else "No Orange Detected"
    orange_status_color = (0, 140, 255) if orange_detected else (0, 0, 255)
    cv2.putText(frame, orange_status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, orange_status_color, 2)

    # Display lime green detection status
    lime_status_text = "Lime Green Detected" if lime_detected else "No Lime Green Detected"
    lime_status_color = (0, 255, 0) if lime_detected else (0, 0, 255)
    cv2.putText(frame, lime_status_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, lime_status_color, 2)

    # Display light blue detection status
    light_blue_status_text = "Light Blue Detected" if light_blue_detected else "No Light Blue Detected"
    light_blue_status_color = (255, 255, 0) if light_blue_detected else (0, 0, 255)
    cv2.putText(frame, light_blue_status_text, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, light_blue_status_color, 2)

    # Write the result frame to the output video if any detection is positive
    if smoothed_detection or orange_detected or lime_detected or light_blue_detected:
        out.write(frame)

    cv2.imshow("Video Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True

# Main loop
while True:
    if not process_video():
        break

# Release the video feed and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
