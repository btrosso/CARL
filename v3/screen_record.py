import subprocess
import os

# Define your recording parameters
screen_index = 2
x = 300
y = 260
width = 1250
height = 650
fps = 30

output_path = os.path.expanduser("~/Documents/screen_recordings/05172025_algo.mp4")
"""
https://www.youtube.com/watch?v=BVS3GELeU3k  
video pt1: start: 00:00 | stopped at 24:17
video pt2: start: 24:18 | stopped at 57:43
"""

# Build ffmpeg command
cmd = [
    "ffmpeg",
    "-f", "avfoundation",
    "-framerate", str(fps),
    "-i", f"{screen_index}:none",
    "-vf", f"crop={width}:{height}:{x}:{y}",
    "-pix_fmt", "yuv420p",  # Ensures QuickTime compatibility
    "-vcodec", "libx264",   # H.264 codec for .mp4
    output_path
]

print("Running ffmpeg...")
print(" ".join(cmd))

# Run the command
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print("Recording failed:", e)
