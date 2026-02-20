import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('videos/Rm cut 2.mp4')

# --- CUSTOM LANE DEFINITIONS ---
lane1 = np.array([[571, 345], [552, 467], [520, 624], [514, 677], [662, 676], [654, 409], [650, 339]], np.int32)
lane2 = np.array([[674, 694], [666, 437], [662, 338], [656, 241], [714, 239], [755, 399], [812, 613], [828, 690]], np.int32)
lane3 = np.array([[839, 694], [998, 689], [876, 421], [783, 235], [756, 200], [703, 199], [748, 354], [793, 506]], np.int32)
lane4 = np.array([[1009, 695], [1183, 701], [982, 425], [892, 299], [821, 312], [888, 427], [973, 600], [1014, 689]], np.int32)

all_lanes = [lane1, lane2, lane3, lane4]

# ====== Settings ======
DWELL_LIMIT_SEC = 3.0
LANE_CHANGE_PAIRS = {(0, 1)}      # Moving from Lane 1 -> Lane 2 is a violation.

# ====== Initialize Lane Masks for Area Overlap ======
success, first_frame = cap.read()
if success:
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
else:
    h, w = 720, 1280 

lane_masks = []
for lane in all_lanes:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lane], 255)
    lane_masks.append(mask)

# ====== State ======
cars_that_entered_lanes = set()
# ONLY tracking the leader for Lane 1 (Index 0) now
lane_leader = {0: {"id": None, "start_time": None}} 
car_last_lane = {}
lane_change_violators = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Draw lanes
    for i, lane in enumerate(all_lanes):
        cv2.polylines(frame, [lane], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(frame, f"Lane {i+1}", tuple(lane[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Detect & Track
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

    if not results or results[0].boxes is None or results[0].boxes.id is None:
        cv2.imshow("Multi-Lane Traffic Timer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # 1) Determine which cars are inside each lane
    cars_in_lane = {i: [] for i in range(len(all_lanes))}

    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        cy = int(y2) 

        # --- AREA OVERLAP LOGIC ---
        current_lane = None
        max_overlap_ratio = 0.0
        bbox_area = max(1, (x2 - x1) * (y2 - y1)) 

        for i, mask in enumerate(lane_masks):
            y1_c, y2_c = max(0, y1), min(h, y2)
            x1_c, x2_c = max(0, x1), min(w, x2)
            
            if y2_c <= y1_c or x2_c <= x1_c:
                continue
                
            roi = mask[y1_c:y2_c, x1_c:x2_c]
            overlap_pixels = cv2.countNonZero(roi)
            overlap_ratio = overlap_pixels / float(bbox_area)
            
            if overlap_ratio > 0.40 and overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio
                current_lane = i

        # --- LANE CHANGE VIOLATION LOGIC ---
        prev_lane = car_last_lane.get(track_id, None)
        
        if prev_lane is not None and current_lane is not None and prev_lane != current_lane:
            if (prev_lane, current_lane) in LANE_CHANGE_PAIRS:
                lane_change_violators.add(track_id)
        
        if current_lane is not None:
            car_last_lane[track_id] = current_lane
            cars_that_entered_lanes.add(track_id)
            cars_in_lane[current_lane].append({
                "id": track_id,
                "cy": cy 
            })

    # 2) Pick the leader ONLY for Lane 1 (Index 0)
    if len(cars_in_lane[0]) == 0:
        lane_leader[0]["id"] = None
        lane_leader[0]["start_time"] = None
    else:
        leader_car = max(cars_in_lane[0], key=lambda d: d["cy"])
        leader_id = leader_car["id"]

        if lane_leader[0]["id"] != leader_id:
            lane_leader[0]["id"] = leader_id
            lane_leader[0]["start_time"] = time.time()

    now = time.time()

    # Fast lookup table
    lane_of_car = {}
    for lane_idx, lst in cars_in_lane.items():
        for d in lst:
            lane_of_car[d["id"]] = lane_idx

    # 3) Draw bounding boxes and labels
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)

        color = (0, 255, 255) # Yellow default
        label = "NO LANE"

        if track_id in lane_of_car:
            lane_idx = lane_of_car[track_id]
            lane_num = lane_idx + 1

            # Check if this specific car is the Lane 1 leader
            is_lane_1_leader = (lane_idx == 0) and (lane_leader[0]["id"] == track_id)

            # --- PRIORITY 1: LANE CHANGE VIOLATOR ---
            if track_id in lane_change_violators:
                color = (0, 0, 255) # RED
                label = f"ILLEGAL LANE CHANGE (L1->L2)"

            # --- PRIORITY 2: LANE 1 LEADER & DWELL TIMER ---
            elif is_lane_1_leader and lane_leader[0]["start_time"] is not None:
                elapsed = now - lane_leader[0]["start_time"]

                if elapsed > DWELL_LIMIT_SEC:
                    color = (0, 0, 255) # RED
                    label = f"L1 {elapsed:.1f}s VIOLATOR"
                else:
                    color = (0, 255, 0) # GREEN
                    label = f"L1 {elapsed:.1f}s LEADER"
            
            # --- PRIORITY 3: ALL OTHER CARS (Lanes 2-4, or L1 non-leaders) ---
            else:
                color = (0, 255, 255) # YELLOW
                label = f"L{lane_num} NORMAL"

        elif track_id in cars_that_entered_lanes:
            color = (100, 100, 100) # GRAY
            label = "EXITED"

        # Draw the box and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Multi-Lane Traffic Timer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()