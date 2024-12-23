import cv2
import os
import time

# Load the Haar cascade classifier for full body detection
body_classifier = cv2.CascadeClassifier("C:\\Users\\user\\Desktop\\Edgematrix\\xml files\\haarcascade_fullbody.xml")

# Check if the classifier loaded successfully
if body_classifier.empty():
    print("Error: Could not load the Haar cascade classifier. Please check the file path.")
    exit()

# Initialize video capture
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a folder to save captured images
save_folder = "captured_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Function to detect bounding boxes for full bodies
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in bodies:
        # Draw a rectangle around detected bodies
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return bodies

# Track the time of the last saved image
last_saved_time = time.time()

# Main loop for video capture and body detection
while True:
    result, video_frame = video_capture.read()
    if not result:
        print("Error: Could not read frame.")
        break

    # Detect full bodies in the frame
    bodies = detect_bounding_box(video_frame)

    # Display the video frame
    cv2.imshow("My Full Body Detection Project", video_frame)

    # Save an image if a body is detected and 2 seconds have passed since the last save
    current_time = time.time()
    if len(bodies) > 0 and (current_time - last_saved_time) >= 2:
        image_path = os.path.join(save_folder, f"capture_{int(current_time)}.jpg")
        cv2.imwrite(image_path, video_frame)
        last_saved_time = current_time

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
