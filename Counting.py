import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for more accuracy

# Initialize DeepSORT tracker with robust settings
tracker = DeepSort(
    max_age=70,  # Increase tracking duration for lost objects
    nn_budget=100,  # Memory budget for feature matching
    embedder="mobilenet",  # Better feature extraction for re-identification
    max_iou_distance=0.7,  # Higher IOU tolerance for tracking re-association
)

# Open video capture
video_path = "C:\\Users\\ASUS TUF\\Downloads\\855565-hd_1920_1080_24fps.mp4"
cap = cv2.VideoCapture(video_path)

seen_ids = {}  # Stores assigned unique IDs
id_counter = 1  # Start ID counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform YOLO inference
    results = model(frame)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            # Detect only persons (YOLO class 0) with high confidence
            if cls == 0 and conf > 0.7:
                detections.append(([x1, y1, x2, y2], conf, cls))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        
        # Maintain consistent ID assignment
        if track_id not in seen_ids:
            seen_ids[track_id] = f'P{id_counter}'  # Assign new unique ID
            id_counter += 1
        
        person_id = seen_ids[track_id]  # Retrieve assigned ID
        
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {person_id}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Debugging Info (Track ID vs Assigned ID)
        print(f"Track ID: {track_id}, Assigned ID: {person_id}")
    
    # Display the total number of detected people
    unique_people_count = len(seen_ids)
    cv2.putText(frame, f'Total People: {unique_people_count}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Show output
    cv2.imshow("People Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
