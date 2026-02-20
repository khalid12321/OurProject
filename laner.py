import cv2

video_path = 'videos/Riyadh Metro 1 cut.mp4'
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}],")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Coordinate Picker", frame)

if success:
    cv2.imshow("Coordinate Picker", frame)
    cv2.setMouseCallback("Coordinate Picker", click_event)
    
    print("--- INSTRUCTIONS ---")
    print("1. Click the corners of your lanes.")
    print("2. Coordinates will print in the terminal.")
    print("3. Press 'ESC' to close this window when finished.")

    while True:
        # Check every 10ms if a key is pressed
        key = cv2.waitKey(10) & 0xFF
        if key == 27: # 27 is the ASCII code for the ESC key
            break
        
        # This allows the window 'X' button to work in some environments
        if cv2.getWindowProperty("Coordinate Picker", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    cap.release()
else:
    print("Error: Could not load video frame.")