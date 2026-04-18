import cv2
import os

"""
Streams frames from all video files in a folder.

Args: 
    folder_path (str): Path to the directory containing video files.

Yields:
    file (str): Video filename
    frame (ndarray): Current frame (BGR format)
    
Usage:
    for name, frame in video_stream("H:/data"):
        # process frame (lane detection, object detection, etc.)
        ...
"""

def video_stream(folder_path):
    for file in os.listdir(folder_path):

        if not file.endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(folder_path, file)
        cam = cv2.VideoCapture(video_path) # open video

        if not cam.isOpened():
            print(f"Failed to open: {file}")
            continue

        print(f"Steaming video: {file}")
        frame_count = 0

        while True:
            # reading the frame
            ret, frame = cam.read()

            if not ret:
                break

            frame_count += 1
            print(f'Reading frame {frame_count}')

            yield file, frame

        cam.release()