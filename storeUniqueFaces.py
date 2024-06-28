import cv2
import face_recognition
import concurrent.futures
import os
import datetime
import numpy as np

print("Starting the script...")

# Function to detect faces in a frame and save unique faces
def process_frame(frame, frame_count, frame_rate, output_dir, saved_face_encodings):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    timestamp = None

    # Check and save each detected face
    for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
        # Check if this face has already been saved
        already_saved = any(face_recognition.compare_faces(saved_face_encodings, face_encoding, tolerance=0.6))
        
        if not already_saved:
            # Save the new face encoding
            saved_face_encodings.append(face_encoding)
            
            # Scale back up face locations since the frame we detected in was scaled to half size
            top, right, bottom, left = face_location
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Save the detected face image
            detected_face = frame[top:bottom, left:right]
            face_filename = os.path.join(output_dir, f"detected_face_{frame_count}_face_{i}.jpg")
            cv2.imwrite(face_filename, detected_face)

        # Scale back up face locations since the frame we detected in was scaled to half size
        top, right, bottom, left = face_location
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw rectangles around the detected faces
        color = (0, 255, 0)  # Green for detected faces
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    return timestamp  # Return the timestamp of the frame

# Create a new output directory for each run
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", run_id)
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Capture frames from a video
video_path = r'C:\Users\tejak\OneDrive\Desktop\vidfacereko\bigexpo5.mp4'
print(f"Video path: {video_path}")
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()
else:
    print("Video file opened successfully")

# Initialize variables
frame_count = 0
frame_rate = cap.get(cv2.CAP_PROP_FPS)
timestamps = []
saved_face_encodings = []

# Skip frames to increase processing speed
skip_frames = 20    # Adjust this value to change how frequently frames are processed
skip_count = 0

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    
    # Break the loop when the video ends
    if not ret:
        print("End of video")
        break

    frame_count += 1

    # Skip frames for faster processing
    skip_count += 1
    if skip_count < skip_frames:
        continue
    skip_count = 0

    # Verify the frame is not empty
    if frame is None or frame.size == 0:
        print("Empty frame")
        continue
    
    # Process each frame using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_frame, frame, frame_count, frame_rate, output_dir, saved_face_encodings)
        timestamp = future.result()
        if timestamp:
            timestamps.append(timestamp)

    # Display the modified frame
    cv2.imshow('Video', frame)

    # Wait for a short period to ensure the display window can refresh
    if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter key
        print("Enter key pressed")
        break

# Print message if no match was found
if not timestamps:
    print("No unique faces saved")
else:
    print("Timestamps of frames with unique faces:", timestamps)

# Release everything if the job is finished
cap.release()
cv2.destroyAllWindows()
print("Released video capture and destroyed all windows")

# Return the timestamps array
timestamps
