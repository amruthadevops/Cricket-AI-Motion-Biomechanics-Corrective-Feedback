# src/utils.py
import cv2
import os

def process_video(video_path, pose_estimator):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_data = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = pose_estimator.extract_pose(frame)
        landmarks = pose_estimator.get_landmarks_array(results)
        
        frames_data.append({
            'frame_number': len(frames_data),
            'landmarks': landmarks,
            'timestamp': len(frames_data) / fps
        })
    
    cap.release()
    return frames_data