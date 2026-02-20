import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8x.pt')
cap = cv2.VideoCapture('your_video_1.mp4')

# --faeeeeeeeeeeeee
lane1 = np.array([[339, 162],
[398, 169],
[415, 209],
[427, 257],
[401, 304],
[356, 357],
[262, 439],
[208, 482],
[140, 538],
[77, 596],
[21, 640],
[253, 643],
[320, 552],
[412, 426],
[495, 316],
[528, 262],
[562, 201],
[569, 175],
[454, 153],
[398, 144],

], np.int32)

lane2 = np.array([[619, 29],
[596, 89],
[574, 160],
[552, 219],
[511, 281],
[458, 373],
[403, 444],
[347, 520],
[265, 627],
[218, 689],
[461, 688],
[510, 572],
[553, 454],
[604, 314],
[629, 216],
[643, 137],
[648, 65],
[657, 28],], np.int32)

lane3 = np.array([[657, 22],
[649, 70],
[640, 148],
[631, 207],
[604, 307],
[573, 403],
[528, 516],
[498, 612],
[461, 688],
[479, 648],
[715, 647],
[714, 548],
[712, 400],
[706, 305],
[704, 246],
[703, 171],
[695, 100],
[693, 66],
[695, 22],], np.int32)

lane4 = np.array([[696, 18],
[694, 73],
[700, 145],
[706, 206],
[707, 302],
[711, 439],
[716, 642],
[961, 647],
[891, 496],
[839, 367],
[799, 273],
[777, 190],
[747, 104],
[736, 50],
[733, 20],

], np.int32)

# Parking polygon (lane 5)
p = np.array([[288, 85],
[339, 87],
[326, 172],
[318, 192],
[247, 190],
], np.int32)

all_lanes = [lane1, lane2, lane3, lane4, p]

# =========================
# Dwell-time thresholds (seconds)
# =========================
# lane index is 1-based here (because current_lane = i + 1)
DWELL_LIMITS = {
    1: 20.0,     # Lane 1: 20 sec
    2: 5.0,      # Lane 2: 5 sec
    3: 5.0,      # Lane 3: 5 sec
    5: 600.0     # Parking (p): 10 minutes = 600 sec
}
# Lane 4 not included => no dwell violation by default

# =========================
# Track state
# =========================
# car_state[track_id] = {"lane": int, "enter_time": float}
car_state = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Draw polygons
    for i, lane in enumerate(all_lanes):
        cv2.polylines(frame, [lane], isClosed=True, color=(255, 255, 0), thickness=2)

        # Keep your original naming style
        cv2.putText(frame, f"Lane {i+1}", tuple(lane[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # Your original point (bottom-center) for polygon test
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            current_lane = None
            for i, lane in enumerate(all_lanes):
                if cv2.pointPolygonTest(lane, (cx, cy), False) >= 0:
                    current_lane = i + 1  # 1..5
                    break

            # If not in any polygon: reset state so next time starts fresh
            if current_lane is None:
                if track_id in car_state:
                    del car_state[track_id]
                continue

            now = time.time()

            # If first time OR lane changed -> reset timer for that lane
            if track_id not in car_state:
                car_state[track_id] = {"lane": current_lane, "enter_time": now}
            else:
                if car_state[track_id]["lane"] != current_lane:
                    car_state[track_id] = {"lane": current_lane, "enter_time": now}

            elapsed = now - car_state[track_id]["enter_time"]

            # Check dwell violation based on the lane-specific limit
            limit = DWELL_LIMITS.get(current_lane, None)  # None => no dwell rule for this lane
            is_violator = (limit is not None) and (elapsed > limit)

            # Colors
            box_color = (0, 0, 255) if is_violator else (0, 255, 0)

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label text
            if current_lane == 5:
                zone_label = "PARKING"
            else:
                zone_label = f"L{current_lane}"

            if is_violator:
                label_text = f"{zone_label} {elapsed:.1f}s VIOLATOR"
            else:
                label_text = f"{zone_label} {elapsed:.1f}s"

            cv2.putText(frame, label_text,
                        (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    cv2.imshow("Multi-Lane Traffic Timer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()