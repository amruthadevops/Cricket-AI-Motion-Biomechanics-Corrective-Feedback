import cv2
import os
import numpy as np
from .visualization import Visualizer
from src.video_processor import CricketVideoProcessor
from .correction_visualizer import CorrectionVisualizer
from src.utils.correction_visualizer import CricketCorrectionVisualizer

class VideoAnnotator(Visualizer):
    def __init__(self, output_path='output/'):
        """
        Initialize the video annotator.
        
        Args:
            output_path: Path to save output files
        """
        super().__init__(output_path)
        
    def annotate_frame_with_feedback(self, frame, keypoints_2d, feedback, score):
        """
        Annotate a video frame with pose feedback.
        
        Args:
            frame: Input video frame
            keypoints_2d: 2D keypoints of the pose
            feedback: Dictionary containing feedback for each joint
            score: Overall score of the pose
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated_frame = frame.copy()
        
        if keypoints_2d is None:
            cv2.putText(annotated_frame, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated_frame
        
        # Draw keypoints
        for name, coords in keypoints_2d.items():
            if isinstance(coords, (list, tuple, np.ndarray)) and len(coords) >= 2:
                # Extract only x and y coordinates
                # Handle different data structures properly
                try:
                    if isinstance(coords, np.ndarray) and coords.ndim > 1:
                        # If it's a multi-dimensional array, take the first element
                        x = int(coords[0][0] * frame.shape[1])
                        y = int(coords[0][1] * frame.shape[0])
                    elif isinstance(coords[0], (list, tuple, np.ndarray)):
                        # If coords[0] is a list, tuple, or array, take its first element
                        x = int(coords[0][0] * frame.shape[1])
                        y = int(coords[1][0] * frame.shape[0])
                    else:
                        # If coords[0] is a scalar
                        x = int(coords[0] * frame.shape[1])
                        y = int(coords[1] * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Error processing coordinates for {name}: {e}")
                    continue
                    
        # Draw connections
        connections = [
            ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        for joint1, joint2 in connections:
            if joint1 in keypoints_2d and joint2 in keypoints_2d:
                coords1 = keypoints_2d[joint1]
                coords2 = keypoints_2d[joint2]
                
                if (isinstance(coords1, (list, tuple, np.ndarray)) and len(coords1) >= 2 and
                    isinstance(coords2, (list, tuple, np.ndarray)) and len(coords2) >= 2):
                    
                    try:
                        # Handle different data structures properly for coords1
                        if isinstance(coords1, np.ndarray) and coords1.ndim > 1:
                            x1 = int(coords1[0][0] * frame.shape[1])
                            y1 = int(coords1[0][1] * frame.shape[0])
                        elif isinstance(coords1[0], (list, tuple, np.ndarray)):
                            x1 = int(coords1[0][0] * frame.shape[1])
                            y1 = int(coords1[1][0] * frame.shape[0])
                        else:
                            x1 = int(coords1[0] * frame.shape[1])
                            y1 = int(coords1[1] * frame.shape[0])
                            
                        # Handle different data structures properly for coords2
                        if isinstance(coords2, np.ndarray) and coords2.ndim > 1:
                            x2 = int(coords2[0][0] * frame.shape[1])
                            y2 = int(coords2[0][1] * frame.shape[0])
                        elif isinstance(coords2[0], (list, tuple, np.ndarray)):
                            x2 = int(coords2[0][0] * frame.shape[1])
                            y2 = int(coords2[1][0] * frame.shape[0])
                        else:
                            x2 = int(coords2[0] * frame.shape[1])
                            y2 = int(coords2[1] * frame.shape[0])
                        
                        cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"Error processing connection between {joint1} and {joint2}: {e}")
                        continue
        
        # Highlight problematic joints
        if feedback:
            for joint, data in feedback.items():
                if joint in keypoints_2d and isinstance(data, dict):
                    coords = keypoints_2d[joint]
                    if isinstance(coords, (list, tuple, np.ndarray)) and len(coords) >= 2:
                        try:
                            # Handle different data structures properly
                            if isinstance(coords, np.ndarray) and coords.ndim > 1:
                                x = int(coords[0][0] * frame.shape[1])
                                y = int(coords[0][1] * frame.shape[0])
                            elif isinstance(coords[0], (list, tuple, np.ndarray)):
                                x = int(coords[0][0] * frame.shape[1])
                                y = int(coords[1][0] * frame.shape[0])
                            else:
                                x = int(coords[0] * frame.shape[1])
                                y = int(coords[1] * frame.shape[0])
                            
                            color = (0, 255, 0) if data.get('status') == 'good' else (0, 0, 255)
                            cv2.circle(annotated_frame, (x, y), 10, color, -1)
                            
                            # Add text annotation
                            joint_name = joint.replace('_', ' ').title()
                            value_text = f"{data.get('value', 0):.1f}°" if 'value' in data else ""
                            cv2.putText(annotated_frame, f"{joint_name}: {value_text}", 
                                    (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        except (IndexError, TypeError, ValueError) as e:
                            print(f"Error highlighting joint {joint}: {e}")
                            continue
        
        # Add overall score
        score_text = f"Score: {score:.1f}%"
        score_color = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 60 else (0, 0, 255)
        cv2.putText(annotated_frame, score_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        return annotated_frame



    
    def create_annotated_video(self, input_video_path, output_video_path, 
                              pose_estimator, feedback_generator, 
                              sample_rate=1):
        """
        Create an annotated video with pose feedback.
        
        Args:
            input_video_path: Path to the input video
            output_video_path: Path to save the output video
            pose_estimator: PoseEstimator object
            feedback_generator: FeedbackGenerator object
            sample_rate: Process every nth frame (to speed up processing)
            
        Returns:
            success: True if video was created successfully
        """
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video file: {input_video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Change output file extension to .avi for better codec compatibility
        if output_video_path.endswith('.mp4'):
            output_video_path = output_video_path[:-4] + '.avi'
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process only every nth frame
            if frame_count % sample_rate != 0:
                continue
                
            processed_count += 1
            
            # Estimate pose
            landmarks, pose_frame = pose_estimator.estimate_pose(frame)
            
            # Get 2D keypoints
            keypoints_2d = pose_estimator.get_2d_keypoints(landmarks)
            
            # Get 3D keypoints
            keypoints_3d = pose_estimator.get_3d_keypoints(landmarks)
            
            # Analyze pose and generate feedback
            feedback, score = feedback_generator.analyze_pose(keypoints_3d)
            
            # Annotate frame
            annotated_frame = self.annotate_frame_with_feedback(
                frame, keypoints_2d, feedback, score
            )
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Print progress
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Video annotation complete. Output saved to {output_video_path}")
        return True


