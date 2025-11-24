"""
Record a portion of the screen (Streamlit app) and save as GIF.
macOS compatible.
"""

import pyautogui
import imageio
import time

# -----------------------------
# CONFIGURATION
# -----------------------------

duration = 10      # total seconds to record
fps = 5            # frames per second
save_path = "streamlit_demo.gif"

# Set the region to capture: (left, top, width, height)
region = None  # capture full screen

# -----------------------------
# RECORD SCREEN
# -----------------------------
frames = []
frame_duration = 1 / fps
total_frames = int(duration * fps)

print(f"Recording {duration} seconds ({total_frames} frames) ...")
time.sleep(2)  # small delay to prepare

for i in range(total_frames):
    screenshot = pyautogui.screenshot(region=region)
    frames.append(screenshot)
    time.sleep(frame_duration)

print("Recording finished. Saving GIF ...")

# -----------------------------
# SAVE GIF
# -----------------------------
frames[0].save(
    save_path,
    save_all=True,
    append_images=frames[1:],
    duration=int(frame_duration*1000),  # duration per frame in ms
    loop=0
)

print(f"GIF saved as {save_path}")
