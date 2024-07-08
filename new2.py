import cv2
import numpy as np
import time
from threading import Thread, Lock

# Global variables to store detection coordinates and a lock to synchronize access
detection_coords = {'x_max': None}
lock = Lock()

def detect_laser_pointer(cap):
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([60, 100, 100])
        upper_green = np.array([65, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Laser Pointer Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
        elapsed_time = time.time() - start_time
        print(f"Laser Pointer Frame {frame_count}: {elapsed_time:.3f}s")
        time.sleep(0.8)

def process_human(cap):
    frame_count = 0
    net = cv2.dnn.readNet("yolov4-csp-s-mish.weights", "yolov4-csp-s-mish.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()
    upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        found_person = False
        unique_persons = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.8 and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x_min = center_x - w // 2
                    y_min = center_y - h // 2
                    x_max = center_x + w // 2
                    y_max = center_y + h // 2
                    person_detected = False
                    for person in unique_persons:
                        px_min, py_min, px_max, py_max = person
                        overlap_area = (min(x_max, px_max) - max(x_min, px_min)) * (min(y_max, py_max) - max(y_min, py_min))
                        box_area = (x_max - x_min) * (y_max - y_min)
                        person_area = (px_max - px_min) * (py_max - py_min)
                        overlap_ratio = overlap_area / float(box_area + person_area - overlap_area)
                        if overlap_ratio > 0.3:
                            person_detected = True
                            break
                    if not person_detected:
                        roi_gray = frame[y_min:y_max, x_min:x_max]
                        upper_bodies = upper_body_cascade.detectMultiScale(roi_gray, 1.1, 2)
                        for (ux, uy, uw, uh) in upper_bodies:
                            cv2.circle(frame, (x_min + ux + uw // 2, y_min + uy + uh // 2), 5, (0, 255, 0), -1)
                            found_person = True
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        unique_persons.append((x_min, y_min, x_max, y_max))
                        if found_person:
                            center_x_3 = str(center_x).zfill(3)
                            center_y_3 = str(center_y).zfill(3)
                            mystring = f"{center_x_3[0]},{center_x_3[1]},{center_x_3[2]},{center_y_3[0]},{center_y_3[1]},{center_y_3[2]}\n"
                            cv2.imshow('Human detection', frame)
                        if found_person:
                            break
                    # Lock access to the shared variable and update it
                    with lock:
                        detection_coords['x_max'] = x_max
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
        elapsed_time = time.time() - start_time
        print(f"Human Frame {frame_count}: {elapsed_time:.3f}s")

# Function to print the x_max value from the shared variable
def print_x_max():
    while True:
        with lock:
            print(f"x_max: {detection_coords['x_max']}")
        time.sleep(1)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    t1 = Thread(target=detect_laser_pointer, args=(cap,))
    t2 = Thread(target=process_human, args=(cap,))
    t3 = Thread(target=print_x_max)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    cap.release()
    cv2.destroyAllWindows()