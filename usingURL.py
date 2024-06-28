import cv2
import face_recognition
import concurrent.futures
from pytube import YouTube
import os

print("Starting the script...")

# Function to download the video from YouTube
def download_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    if stream:
        output_path = stream.download()
        return output_path
    else:
        print("Error: No suitable video stream found.")
        return None

# Function to detect faces in a frame and compare them with the reference face encoding
def process_frame(frame, reference_face_encoding, frame_count, frame_rate):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    matched = False
    timestamp = None

    # Compare each detected face with the reference face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        match = face_recognition.compare_faces([reference_face_encoding], face_encoding, tolerance=0.6)
        
        if match[0]:
            timestamp_seconds = frame_count / frame_rate
            timestamp_minutes = int(timestamp_seconds // 60)
            timestamp_seconds %= 60
            timestamp = (timestamp_minutes, timestamp_seconds)
            print(f"Match found at {timestamp_minutes} minutes {timestamp_seconds} seconds")
            matched = True

        # Scale back up face locations since the frame we detected in was scaled to half size
        top, right, bottom, left = face_location
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw rectangles around the detected faces
        color = (0, 0, 255) if match[0] else (0, 255, 0)  # Red if matched, green otherwise
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    return matched, timestamp  # Return whether a match was found in this frame and the timestamp

# Main function to process the video
def process_video(video_url):
    if "youtube.com" in video_url or "youtu.be" in video_url:
        # Download the video if it's a YouTube URL
        print("Downloading video from YouTube...")
        video_path = download_video(video_url)
        if not video_path:
            print("Failed to download the video.")
            return
    else:
        video_path = video_url

    print(f"Video path: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    else:
        print("Video file opened successfully")

    # Load the reference image
    reference_image_path = r'C:\Users\tejak\OneDrive\Desktop\vidfacereko\reference_image4.jpg'
    reference_image = face_recognition.load_image_file(reference_image_path)

    # Extract face encoding from the reference image
    reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

    # Initialize variables
    frame_count = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    match_found = False
    timestamps = []

    # Skip frames to increase processing speed
    skip_frames = 25    # Adjust this value to change how frequently frames are processed
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
            future = executor.submit(process_frame, frame, reference_face_encoding, frame_count, frame_rate)
            match, timestamp = future.result()
            if match:
                match_found = True
                timestamps.append(timestamp)

        # Display the modified frame
        cv2.imshow('Video', frame)

        # Wait for a short period to ensure the display window can refresh
        if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter key
            print("Enter key pressed")
            break

    # Print message if no match was found
    if not match_found:
        print("Match not found")
    else:
        print("Timestamps of matches:", timestamps)

    # Release everything if the job is finished
    cap.release()
    cv2.destroyAllWindows()
    print("Released video capture and destroyed all windows")

    # Return the timestamps array
    return timestamps

# Example usage
video_url = 'https://youtu.be/0buHlSH-W2Q?si=GKPSYExQkAsfRWVi'  # Replace with your video URL
timestamps = process_video(video_url)
print("Timestamps of matches:", timestamps)
