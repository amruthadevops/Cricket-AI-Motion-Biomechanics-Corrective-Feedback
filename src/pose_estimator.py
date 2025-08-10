# src/pose_estimator.py
import cv2
import mediapipe as mp
import numpy as np

# =============================================================================
# POSE ESTIMATOR CLASS
# =============================================================================

class CricketPoseEstimator:
    def __init__(self):
        # CPU-optimized settings for basic hardware
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use lighter model
            enable_segmentation=False,  # Disable to save computation
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define important cricket joints
        self.cricket_joints = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
    
    def extract_pose(self, frame):
        """Extract pose from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
    
    def get_landmarks_array(self, results):
        """Convert landmarks to numpy array"""
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(landmarks)
        return None
    
    def get_key_points(self, results):
        """Extract only important cricket joints"""
        if results.pose_landmarks:
            key_points = {}
            for joint_name, joint_idx in self.cricket_joints.items():
                landmark = results.pose_landmarks.landmark[joint_idx]
                key_points[joint_name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            return key_points
        return None
    
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
                self.mp_draw.draw_landmarks(
                    pose_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Extract landmarks as numpy array
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
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
            landmarks: MediaPipe pose landmarks or numpy array
            
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
            
            # Check if landmarks is a numpy array (already converted)
            if isinstance(landmarks, np.ndarray):
                # If it's a numpy array, extract coordinates directly
                for name, idx in key_point_indices.items():
                    if idx * 3 < len(landmarks):  # Each landmark has x,y,z
                        x = landmarks[idx * 3]
                        y = landmarks[idx * 3 + 1]
                        keypoints[name] = [x, y]
            else:
                # If it's MediaPipe landmarks, extract coordinates using .x, .y attributes
                for name, idx in key_point_indices.items():
                    if idx < len(landmarks):
                        point = landmarks[idx]
                        keypoints[name] = [point.x, point.y]
            
            return keypoints
        except Exception as e:
            print(f"Error extracting 2D keypoints: {e}")
            return None



    def get_3d_keypoints(self, landmarks):
        """
        Extract 3D keypoints from landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks or numpy array
            
        Returns:
            keypoints: Dictionary of 3D keypoints or None if no landmarks
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
            
            # Check if landmarks is a numpy array (already converted)
            if isinstance(landmarks, np.ndarray):
                # If it's a numpy array, extract coordinates directly
                for name, idx in key_point_indices.items():
                    if idx * 3 < len(landmarks):  # Each landmark has x,y,z
                        x = landmarks[idx * 3]
                        y = landmarks[idx * 3 + 1]
                        z = landmarks[idx * 3 + 2]
                        keypoints[name] = [x, y, z]
            else:
                # If it's MediaPipe landmarks, extract coordinates using .x, .y, .z attributes
                for name, idx in key_point_indices.items():
                    if idx < len(landmarks):
                        point = landmarks[idx]
                        keypoints[name] = [point.x, point.y, point.z]
            
            return keypoints
        except Exception as e:
            print(f"Error extracting 3D keypoints: {e}")
            return None

