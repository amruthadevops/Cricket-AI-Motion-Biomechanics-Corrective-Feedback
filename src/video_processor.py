import cv2
import numpy as np
from src.pose_estimator import CricketPoseEstimator
import os
import mediapipe as mp


class CricketVideoProcessor:
    def __init__(self, skip_frames=2):
        self.pose_estimator = CricketPoseEstimator()
        self.skip_frames = skip_frames

    # In the process_video method, remove the frequent print statements
    # and only keep essential logging
    
    def process_video(self, video_path, max_frames=200):
        """Process video and return consistent frame data."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {video_path}")
            return []
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video info: {total_frames} frames at {fps} FPS")
    
        frame_data = []
        count, processed = 0, 0
    
        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
    
            # Skip frames based on skip_frames setting
            if count % (self.skip_frames + 1) != 0:
                count += 1
                continue
    
            # Resize frame if too large for processing
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
            # Extract pose data
            try:
                results = self.pose_estimator.extract_pose(frame)
                landmarks = self.pose_estimator.get_landmarks_array(results)
                keypoints = self.pose_estimator.get_key_points(results)
    
                # Ensure landmarks are properly formatted
                pose_detected = landmarks is not None
                if pose_detected and isinstance(landmarks, np.ndarray):
                    # Ensure landmarks are in the correct shape
                    if landmarks.ndim == 2 and landmarks.shape == (33, 4):
                        # Already in correct shape, store flattened version for consistency
                        landmarks_2d = landmarks
                        landmarks_flat = landmarks.flatten()
                    elif landmarks.size == 132:  # 33 * 4
                        # Reshape from flat to 2D
                        landmarks_2d = landmarks.reshape(33, 4)
                        landmarks_flat = landmarks.flatten()
                    else:
                        # Invalid size, mark as no pose detected
                        pose_detected = False
                        landmarks_2d = None
                        landmarks_flat = None
                else:
                    landmarks_2d = None
                    landmarks_flat = None
    
                frame_info = {
                    'frame_number': count,
                    'timestamp': count / fps if fps > 0 else count * 0.033,  # fallback to 30fps estimate
                    'landmarks': landmarks_2d,  # Store as 2D array (33, 4)
                    'landmarks_flat': landmarks_flat,  # Store flattened version for compatibility
                    'key_points': keypoints,
                    'pose_detected': pose_detected,
                    'original_shape': (h, w),
                    'processed_shape': frame.shape[:2]
                }
    
                frame_data.append(frame_info)
                processed += 1
    
            except Exception as e:
                # Still add frame data but with no pose
                frame_info = {
                    'frame_number': count,
                    'timestamp': count / fps if fps > 0 else count * 0.033,
                    'landmarks': None,
                    'landmarks_flat': None,
                    'key_points': None,
                    'pose_detected': False,
                    'original_shape': (h, w),
                    'processed_shape': frame.shape[:2],
                    'error': str(e)
                }
                frame_data.append(frame_info)
                processed += 1
    
            count += 1
    
        cap.release()
        print(f"Completed processing: {processed} frames")
        
        return frame_data

    def extract_keypoints_sequences(self, frame_data):
        """
        Extract keypoint sequences for compatibility with legacy analyzers.
        Returns: (keypoints_2d, keypoints_3d, frame_times)
        """
        keypoints_2d = []
        keypoints_3d = []
        frame_times = []
        
        for frame in frame_data:
            if frame.get('pose_detected') and frame.get('landmarks') is not None:
                landmarks = frame['landmarks']
                
                # Extract 2D keypoints (x, y coordinates)
                if isinstance(landmarks, np.ndarray) and landmarks.shape == (33, 4):
                    keypoints_2d.append(landmarks[:, :2])  # x, y only
                    # For 3D, we'll use x, y, z (assuming z is visibility or depth)
                    keypoints_3d.append(landmarks[:, :3])  # x, y, z
                else:
                    # Fallback - create zero arrays
                    keypoints_2d.append(np.zeros((33, 2)))
                    keypoints_3d.append(np.zeros((33, 3)))
                
                frame_times.append(frame.get('timestamp', 0))
            else:
                # Add zero arrays for frames without pose
                keypoints_2d.append(np.zeros((33, 2)))
                keypoints_3d.append(np.zeros((33, 3)))
                frame_times.append(frame.get('timestamp', 0))
        
        return (
            np.array(keypoints_2d) if keypoints_2d else np.zeros((0, 33, 2)),
            np.array(keypoints_3d) if keypoints_3d else np.zeros((0, 33, 3)),
            np.array(frame_times) if frame_times else np.array([])
        )

    def get_frame_statistics(self, frame_data):
        """Get statistics about the processed frames."""
        if not frame_data:
            return {}
            
        total_frames = len(frame_data)
        valid_poses = sum(1 for f in frame_data if f.get('pose_detected', False))
        error_frames = sum(1 for f in frame_data if 'error' in f)
        
        # Calculate average confidence if available
        confidences = []
        for frame in frame_data:
            if frame.get('landmarks') is not None:
                landmarks = frame['landmarks']
                if isinstance(landmarks, np.ndarray) and landmarks.shape == (33, 4):
                    # Average visibility/confidence scores
                    confidences.extend(landmarks[:, 3])
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_frames': total_frames,
            'valid_poses': valid_poses,
            'error_frames': error_frames,
            'pose_detection_rate': valid_poses / total_frames if total_frames > 0 else 0,
            'average_confidence': avg_confidence,
            'processing_success_rate': (total_frames - error_frames) / total_frames if total_frames > 0 else 0
        }
        
    def estimate_pose(self, image):
        """
        Estimate pose from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            tuple: (landmarks, pose_image)
                landmarks: numpy array of pose landmarks or None if no pose detected
                pose_image: image with pose drawn on it
        """
        try:
            # Convert the BGR image to RGB before processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and get the pose landmarks
            results = self.pose.process(image_rgb)
            
            # Create a copy of the image to draw on
            pose_image = image.copy()
            
            # Draw the pose annotation on the image
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    pose_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Extract landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                return np.array(landmarks), pose_image
            else:
                return None, pose_image
                
        except Exception as e:
            print(f"Error estimating pose: {e}")
            return None, image

    def get_2d_keypoints(self, landmarks):
        """
        Extract 2D keypoints from landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            keypoints: Dictionary of 2D keypoints or None if no landmarks
        """
        if landmarks is None:
            return None
            
        try:
            # Extract key points
            keypoints = {}
            
            # Define key point indices
            key_point_indices = {
                'nose': 0,
                'left_eye': 1,
                'right_eye': 2,
                'left_ear': 3,
                'right_ear': 4,
                'left_shoulder': 11,
                'right_shoulder': 12,
                'left_elbow': 13,
                'right_elbow': 14,
                'left_wrist': 15,
                'right_wrist': 16,
                'left_hip': 23,
                'right_hip': 24,
                'left_knee': 25,
                'right_knee': 26,
                'left_ankle': 27,
                'right_ankle': 28
            }
            
            # Extract key points
            for name, idx in key_point_indices.items():
                if idx < len(landmarks):
                    point = landmarks[idx]
                    keypoints[name] = [point.x, point.y]
            
            return keypoints
        except Exception as e:
            print(f"Error extracting 2D keypoints: {e}")
            return None
