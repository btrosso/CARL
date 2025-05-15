import subprocess
import os

# Define your recording parameters
screen_index = 2
x = 300
y = 260
width = 1250
height = 650
fps = 30

output_path = os.path.expanduser("~/Documents/screen_recordings/05142025.mp4")

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
