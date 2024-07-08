import cv2
import numpy as np
#import serial
import time

#arduino_x = serial.Serial(port='COM9', baudrate=9600, timeout=1)



def detect_laser_pointer(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([60, 100, 100])
    upper_green = np.array([65, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply Gaussian Blur to reduce noise
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def send_string_to_arduino(string):
    arduino_x.write(string.encode())

net = cv2.dnn.readNet("yolov4-csp-s-mish.weights", "yolov4-csp-s-mish.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()
upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

cap = cv2.VideoCapture("12.mp4")
frame_count = 0

def process_frame(frame, net, output_layers, upper_body_cascade):
    height, width, channels = frame.shape
    
    # Create blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    found_person = False

    # Lists to store unique detected persons
    unique_persons = []

    # Process YOLO model output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if detected object is a person with high confidence
            if confidence > 0.8 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate box coordinates
                x_min = center_x - w // 2
                y_min = center_y - h // 2
                x_max = center_x + w // 2
                y_max = center_y + h // 2

                # Check if this person overlaps with any already detected persons
                person_detected = False
                for person in unique_persons:
                    px_min, py_min, px_max, py_max = person
                    overlap_area = (min(x_max, px_max) - max(x_min, px_min)) * (min(y_max, py_max) - max(y_min, py_min))
                    box_area = (x_max - x_min) * (y_max - y_min)
                    person_area = (px_max - px_min) * (py_max - py_min)
                    overlap_ratio = overlap_area / float(box_area + person_area - overlap_area)

                    # If overlap ratio is greater than threshold, consider it the same person
                    if overlap_ratio > 0.3:
                        person_detected = True
                        break

                if not person_detected:
                    # Extract ROI for upper body detection
                    roi_gray = frame[y_min:y_max, x_min:x_max]
                    upper_bodies = upper_body_cascade.detectMultiScale(roi_gray, 1.1, 2)

                    # Draw circles and rectangles for detected upper bodies
                    for (ux, uy, uw, uh) in upper_bodies:
                        cv2.circle(frame, (x_min + ux + uw // 2, y_min + uy + uh // 2), 5, (0, 255, 0), -1)
                        found_person = True

                    # Draw bounding box around the detected person
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                    # Store unique person
                    unique_persons.append((x_min, y_min, x_max, y_max))

                    # If person is found, format string for Arduino (optional)
                    if found_person:
                        center_x_3 = str(center_x).zfill(3)
                        center_y_3 = str(center_y).zfill(3)
                        mystring = f"{center_x_3[0]},{center_x_3[1]},{center_x_3[2]},{center_y_3[0]},{center_y_3[1]},{center_y_3[2]}\n"
                        #send_string_to_arduino(mystring)

                    # Early exit if a person is found to optimize further detections
                    if found_person:
                        break

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()

    frame = detect_laser_pointer(frame)
    final_frame = process_frame(frame, net, output_layers, upper_body_cascade) 
    cv2.imshow('Laser Pointer Detection', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    print(f"Frame {frame_count}: {elapsed_time:.3f}s")

#arduino_x.close()
cap.release()
cv2.destroyAllWindows()

