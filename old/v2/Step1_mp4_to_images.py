import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=1, crop_box=None):
    """
    Extracts frames from an MP4 video and saves them as images with an optional cropping box.

    Args:
        video_path (str): Path to the input MP4 video file.
        output_dir (str): Directory where extracted frames will be saved.
        frame_interval (int): Extract every 'n' frames (default is 1, meaning every frame).
        crop_box (tuple or None): (x, y, width, height) to crop a specific region in each frame. Default is None (saves full frame).

    Returns:
        int: Number of frames saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return 0

    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}")
    print(f"Number of Frames to be Saved: {total_frames // frame_interval}")
    input("To Continue Processing, press 'Enter': ")
    
    frame_count = 0
    saved_frames = 0

    # Use tqdm for progress tracking
    with tqdm(total=total_frames, desc="Extracting Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Save frame every 'frame_interval' frames
            if frame_count % frame_interval == 0:
                if crop_box:
                    x, y, w, h = crop_box
                    frame = frame[y:y+h, x:x+w]  # Crop the frame

                frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frames += 1

            frame_count += 1
            pbar.update(1)  # Update progress bar

    # Release video capture
    cap.release()

    print(f"Extraction complete. {saved_frames} frames saved to {output_dir}.")
    return saved_frames

if __name__ == '__main__':
    video_path = '/Users/brosso/Documents/screen_recordings/youtube_video_traffic.mov'
    output_dir = '/Users/brosso/Documents/personal_code/CARL/data2/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # extract_frames(video_path, output_dir, frame_interval=2, crop_box=(700, 200, 408, 360))
    extract_frames(video_path, output_dir, frame_interval=15)