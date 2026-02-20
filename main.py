import cv2
import time
import math
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# --- CHANGED: INPUT SOURCE IS NOW A VIDEO FILE ---
video_path = 'videos/Relaxing highway traffic.mp4'
cap = cv2.VideoCapture(video_path)


PIXELS_PER_METER = 100  
SPEED_THRESHOLD_KMH = 15 

#TESTTTTTTTTTTTTTTTTTTTTTT
lane1 = np.array([[859, 486],
[836, 667],
[776, 971],
[767, 1044],
[984, 1040],
[979, 484],], np.int32)

lane2 = np.array([[979, 401],
[984, 548],
[994, 1027],
[1244, 1031],
[1177, 751],
[1118, 523],
[1074, 362],
[976, 356],], np.int32)

lane3 = np.array([[859, 486],
[836, 667],
[776, 971],
[767, 1044],
[984, 1040],
[979, 484],], np.int32)

all_lanes = [lane1]
car_timers = {}
cars_that_entered_lanes = set() 
track_history = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        # Loop the video if it ends (optional)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    for i, lane in enumerate(all_lanes):
        cv2.polylines(frame, [lane], isClosed=True, color=(255, 255, 0), thickness=2)

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        current_time = time.time()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int(y2) 

            # --- 1. CALCULATE SPEED ---
            speed_kmh = 0.0
            if track_id in track_history:
                prev_x, prev_y, prev_time = track_history[track_id]
                dist_pixels = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                time_diff = current_time - prev_time
                
                if time_diff > 0:
                    speed_px_s = dist_pixels / time_diff
                    speed_mps = speed_px_s / PIXELS_PER_METER
                    speed_kmh = speed_mps * 3.6

            track_history[track_id] = (cx, cy, current_time)

            # --- 2. CHECK LANE & TIMER LOGIC ---
            current_lane = None
            for i, lane in enumerate(all_lanes):
                if cv2.pointPolygonTest(lane, (cx, cy), False) >= 0:
                    current_lane = i + 1
                    break

            color = (0, 255, 0) # Green
            label = "NO LANE"

            if current_lane:
                cars_that_entered_lanes.add(track_id)
                
                # Check if slow/stopped
                if speed_kmh < SPEED_THRESHOLD_KMH:
                    if track_id not in car_timers:
                        car_timers[track_id] = time.time()
                    
                    elapsed = time.time() - car_timers[track_id]
                    
                    if elapsed > 3.0:
                        color = (0, 0, 255) # Red (Violation)
                    
                    # Show speed in label
                    label = f"STOPPED {elapsed:.1f}s ({int(speed_kmh)}km/h)"
                else:
                    # Reset timer if they start moving again
                    if track_id in car_timers:
                        del car_timers[track_id]
                    label = f"MOVING ({int(speed_kmh)}km/h)"

            elif track_id in cars_that_entered_lanes:
                color = (0, 0, 0) # Black
                label = "EXITED"
                if track_id in car_timers:
                    del car_timers[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Multi-Lane Traffic Timer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
##