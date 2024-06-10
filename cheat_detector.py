import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder):
    
    # Check if the output path already exists
    if not os.path.exists(output_folder):
        # If not, create a new directory
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Loop through the frames in the video
    while cap.isOpened():

        # Read the video frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create the frame path with a folder and the frame count
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame) # Write frame to path
        frame_count += 1

    # Release the capture
    cap.release()

    print(f"Extracted {frame_count} frames from {video_path}.")

def detect_cheat(template_path, frames_folder):

    # Define template
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1] # Width and height of the template

    # Loop through each frame from the video
    for frame_file in os.listdir(frames_folder):

        # Create the frame path
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path, 0)

        # Perform template matching to find the template in the frame
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold) # Locations where the match is above the threshold

        # Check if any matches are found
        if len(loc[0]) > 0:
            print(f"Cheat detected in frame {frame_file}.")
            break
    else:
        print("No cheat detected :)")

def detect_speed_cheat(video_path, threshold=2.0):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Read the first frame
    ret, frame1 = cap.read()

    # Check if the first frame was able to be read
    if not ret:
        print("Error: Could not read the first frame of the video")
        return

    # Convert frame to grayscale to process easier
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Loop through the video frames
    while cap.isOpened():

        ret, frame2 = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Calculate the optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate the direction and magnitude of motion
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Get the average magnitude for each frame
        avg_magnitude = np.mean(magnitude)

        print(f"Average magnitude: {avg_magnitude}")  # Debug print
        
        if avg_magnitude > threshold:
            print("Speed cheat detected.")
            break
        
        # Update the last frame to the current frame
        prev_gray = gray
    
    cap.release()

    print("No speed cheat detected.")


# Detect speed cheat in the video
detect_speed_cheat('No Cheat - Doom Test 1.mp4')