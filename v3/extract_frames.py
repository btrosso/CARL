import cv2
import os

# === CONFIG ===
# video_path = "/Users/brosso/Documents/screen_recordings/youtube_video_traffic.mov"         # ← Replace with your file
video_path = "/Users/brosso/Documents/screen_recordings/output3.mp4"         # ← Replace with your file
# output_dir = "/Users/brosso/Documents/personal_code/CARL/yt_traffic_high_quality/"     # ← Output folder
output_dir = "/Users/brosso/Documents/personal_code/CARL/algotraffic_low_qual/"     # ← Output folder
frame_interval = 100                    # ← Extract 1 frame every 100 frames

# === SETUP ===
os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
frame_count = 0
candidate_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if saved_count == 1001:  # ← Early stop the script if you want to control total number of imgs for now
        break
    if frame_count % frame_interval == 0:
        if candidate_count < 500:
            filename = os.path.join(output_dir, f"frame_{saved_count:05}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        else:
            candidate_count += 1
    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames to {output_dir}")
