import cv2
import numpy as np
import openvino.runtime as ov

def load_models():
    # Load both person detection and pose estimation models
    core = ov.Core()
    person_detection_model = core.compile_model(model="models/person-detection-0200.xml", device_name="CPU")
    pose_estimation_model = core.compile_model(model="models/human-pose-estimation-0001.xml", device_name="CPU")
    return person_detection_model, pose_estimation_model

def process_frame(frame, person_model, pose_model, alert_threshold):
    # Preprocess frame for person detection
    person_input_blob = cv2.resize(frame, (640, 480))
    person_input_blob = person_input_blob.transpose((2, 0, 1))  # HWC to CHW format
    person_input_blob = person_input_blob.reshape(1, 3, 480, 640)

    # Person detection
    person_results = person_model([person_input_blob])[person_model.output(0)]

    # Process each detected person
    fall_detected = False
    for detection in person_results:
        _, label, confidence, xmin, ymin, xmax, ymax = detection
        if confidence > alert_threshold:
            # Extract the person’s region and perform pose estimation
            person_region = frame[int(ymin * frame.shape[0]):int(ymax * frame.shape[0]),
                                  int(xmin * frame.shape[1]):int(xmax * frame.shape[1])]

            # Preprocess the person region for pose estimation
            pose_input_blob = cv2.resize(person_region, (256, 256))
            pose_input_blob = pose_input_blob.transpose((2, 0, 1))
            pose_input_blob = pose_input_blob.reshape(1, 3, 256, 256)

            # Pose estimation inference
            pose_results = pose_model([pose_input_blob])[pose_model.output(0)]

            # Check if the pose results indicate a fall (simple criteria)
            if is_fall(pose_results):
                fall_detected = True
                cv2.rectangle(frame, (int(xmin * frame.shape[1]), int(ymin * frame.shape[0])),
                              (int(xmax * frame.shape[1]), int(ymax * frame.shape[0])), (0, 0, 255), 2)  # Red box for fall
            else:
                cv2.rectangle(frame, (int(xmin * frame.shape[1]), int(ymin * frame.shape[0])),
                              (int(xmax * frame.shape[1]), int(ymax * frame.shape[0])), (0, 255, 0), 2)  # Green box otherwise

    # Display alert if a fall is detected
    if fall_detected:
        send_alert()

    return frame

def is_fall(pose_results):
    """
    Determine if a fall has occurred based on pose estimation results.
    This function analyzes the key points detected by the pose estimation model.
    
    Args:
        pose_results: Output from the pose estimation model.
        
    Returns:
        bool: True if a fall is detected, otherwise False.
    """
    # Basic criteria for fall detection: head and torso key points are close to the ground level.
    # Check if the coordinates of the head are significantly lower than usual.
    
    keypoints = pose_results[0]  # Extract key points from the model output
    head_y = keypoints[0][1]  # Example: Head Y coordinate
    torso_y = keypoints[1][1]  # Example: Torso Y coordinate
    
    # Simple rule: if head and torso are at lower positions than expected, classify as a fall.
    if head_y > 0.8 and torso_y > 0.8:  # Assuming normalized coordinates
        return True
    return False

def send_alert():
    """
    Function to send alert notifications when an incident is detected.
    """
    # Placeholder for integrating alert systems (SMS, email, app notifications)
    print("Alert: Fall detected. Notifying caregivers.")
