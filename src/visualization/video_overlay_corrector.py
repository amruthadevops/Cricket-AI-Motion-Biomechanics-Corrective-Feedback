# src/visualization/video_overlay_corrector.py
import traceback
import cv2
from matplotlib import pyplot as plt
import numpy as np
from .visualization import Visualizer
from .correction_visualizer import CorrectionVisualizer
from src.utils.keypoints_utils import normalize_keypoints, validate_keypoints


class VideoOverlayCorrector(CorrectionVisualizer):
    def __init__(self, output_path='None'):
        """
        Initialize the video overlay corrector.
        
        Args:
            output_path: Path to save output files
        """
        super().__init__(output_path)
        
    def create_overlay_video(self, input_video_path, output_video_path,
                               pose_estimator, feedback_generator, 
                               ideal_keypoints_3d, sample_rate=2):
        """
        Create a video with 3D pose overlay.
        
        Args:
            input_video_path: Path to the input video
            output_video_path: Path to save the output video
            pose_estimator: PoseEstimator object
            feedback_generator: FeedbackGenerator object
            ideal_keypoints_3d: 3D keypoints of the ideal pose
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
            
            # Get 3D keypoints
            keypoints_3d = pose_estimator.get_3d_keypoints(landmarks)
            
            # Generate feedback
            feedback, score = feedback_generator.analyze_pose(keypoints_3d)
            
            # Generate corrections
            corrections = self._generate_corrections(keypoints_3d, ideal_keypoints_3d, feedback)
            
            # Create visualization
            fig = self.create_3d_plot(
                keypoints_3d, feedback, f"3D Pose Analysis (Frame {processed_count})"
            )
            
            # Convert to image
            overlay_img = self.plot_to_image(fig)
            
            # Resize overlay to match video frame
            overlay_img = cv2.resize(overlay_img, (width, height))
            
            # Blend overlay with original frame
            alpha = 0.7  # Transparency factor
            gamma = 0    # Gamma correction
            blended_frame = cv2.addWeighted(frame, 1 - alpha, overlay_img, alpha, gamma)
            
            # Write frame to output video
            out.write(blended_frame)
            
            # Print progress
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Video overlay complete. Output saved to {output_video_path}")
        return True

    def _generate_corrections(self, keypoints_3d_current, keypoints_3d_ideal, feedback):
        """
        Generate correction vectors between current and ideal poses.
        
        Args:
            keypoints_3d_current: 3D keypoints of the current pose
            keypoints_3d_ideal: 3D keypoints of the ideal pose
            feedback: Dictionary containing feedback for each joint
            
        Returns:
            corrections: Dictionary containing correction vectors for each joint
        """
        corrections = {}
        
        # Check if we have valid keypoints
        if keypoints_3d_current is None or keypoints_3d_ideal is None:
            return corrections
        
        # Convert to numpy arrays
        keypoints_current = self._convert_keypoints_to_array(keypoints_3d_current)
        keypoints_ideal = self._convert_keypoints_to_array(keypoints_3d_ideal)
        
        if keypoints_current is None or keypoints_ideal is None:
            return corrections
        
        # Map joint names to keypoint indices
        joint_indices = {
            'right_elbow': 14,
            'right_knee': 26,
            'right_shoulder': 12,
            'right_hip': 24,
            'spine': 11  # Using shoulder as representative for spine
        }
        
        # Generate corrections for each joint
        for joint_name, idx in joint_indices.items():
            if idx < len(keypoints_current) and idx < len(keypoints_ideal):
                # Get current and ideal positions
                current_pos = keypoints_current[idx]
                ideal_pos = keypoints_ideal[idx]
                
                # Calculate correction vector
                correction = ideal_pos - current_pos
                
                # Add to corrections dictionary
                corrections[joint_name] = correction
        
        return corrections

    def create_3d_overlay_video(self, input_video_path, output_video_path,
                               pose_estimator, feedback_generator, 
                               ideal_keypoints_3d, sample_rate=2):
        """
        Create a video with 3D overlay using addWeighted with proper parameters.
        
        Args:
            input_video_path: Path to the input video
            output_video_path: Path to save the output video
            pose_estimator: PoseEstimator object
            feedback_generator: FeedbackGenerator object
            ideal_keypoints_3d: 3D keypoints of the ideal pose
            sample_rate: Process every nth frame (to speed up processing)
            
        Returns:
            success: True if video was created successfully
        """
        try:
            return self.create_overlay_video(
                input_video_path, output_video_path, pose_estimator, 
                feedback_generator, ideal_keypoints_3d, sample_rate
            )
        except Exception as e:
            print(f"⚠️ Error creating 3D overlay video: {e}")
            return False

    def create_side_by_side_video(self, input_video_path, output_video_path,
                                  pose_estimator, feedback_generator, 
                                  ideal_keypoints_3d, sample_rate=2):
        """
        Create a side-by-side comparison video.
        
        Args:
            input_video_path: Path to the input video
            output_video_path: Path to save the output video
            pose_estimator: PoseEstimator object
            feedback_generator: FeedbackGenerator object
            ideal_keypoints_3d: 3D keypoints of the ideal pose
            sample_rate: Process every nth frame (to speed up processing)
            
        Returns:
            success: True if video was created successfully
        """
        try:
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
            
            # Create video writer with double width for side-by-side
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))
            
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
                
                # Get 3D keypoints
                keypoints_3d = pose_estimator.get_3d_keypoints(landmarks)
                
                # Generate feedback
                feedback, score = feedback_generator.analyze_pose(keypoints_3d)
                
                # Create visualization
                fig = self.create_3d_plot(
                    keypoints_3d, feedback, f"3D Pose Analysis (Frame {processed_count})"
                )
                
                # Convert to image
                overlay_img = self.plot_to_image(fig)
                overlay_img = cv2.resize(overlay_img, (width, height))
                
                # Create side-by-side frame
                side_by_side = np.hstack((frame, overlay_img))
                
                # Write frame to output video
                out.write(side_by_side)
                
                # Print progress
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames...")
            
            # Release resources
            cap.release()
            out.release()
            
            print(f"Side-by-side video complete. Output saved to {output_video_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error creating comparison video: {e}")
            return False

    def create_correction_video(self, input_video_path, output_video_path,
                               pose_estimator, feedback_generator, 
                               ideal_keypoints_3d, sample_rate=2):
        """
        Create a correction video showing pose improvements.
        
        Args:
            input_video_path: Path to the input video
            output_video_path: Path to save the output video
            pose_estimator: PoseEstimator object
            feedback_generator: FeedbackGenerator object
            ideal_keypoints_3d: 3D keypoints of the ideal pose
            sample_rate: Process every nth frame (to speed up processing)
            
        Returns:
            success: True if video was created successfully
        """
        try:
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
                
                # Get 3D keypoints
                keypoints_3d = pose_estimator.get_3d_keypoints(landmarks)
                
                # Generate feedback and corrections
                feedback, score = feedback_generator.analyze_pose(keypoints_3d)
                corrections = self._generate_corrections(keypoints_3d, ideal_keypoints_3d, feedback)
                
                # Create corrected pose visualization
                fig = self.create_corrected_pose_visualization(
                    keypoints_3d, ideal_keypoints_3d, feedback, corrections,
                    f"Pose Correction (Frame {processed_count})"
                )
                
                # Convert to image
                correction_img = self.plot_to_image(fig)
                correction_img = cv2.resize(correction_img, (width, height))
                
                # Write frame to output video
                out.write(correction_img)
                
                # Print progress
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames...")
            
            # Release resources
            cap.release()
            out.release()
            
            print(f"Correction video complete. Output saved to {output_video_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error creating correction video: {e}")
            return False
    
    def create_correction_animation_video(self, input_video_path, output_video_path,
                                        pose_estimator, feedback_generator, 
                                        ideal_keypoints_3d, sample_rate=2, animation_frames=10):
        """
        Create a video with animation showing the transition from current to ideal pose.
        
        Args:
            input_video_path: Path to the input video
            output_video_path: Path to save the output video
            pose_estimator: PoseEstimator object
            feedback_generator: FeedbackGenerator object
            ideal_keypoints_3d: 3D keypoints of the ideal pose
            sample_rate: Process every nth frame (to speed up processing)
            animation_frames: Number of frames to use for the animation
            
        Returns:
            success: True if video was created successfully
        """
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_video_path}")
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
            
            # Get 3D keypoints
            keypoints_3d = pose_estimator.get_3d_keypoints(landmarks)
            
            # Generate animation frames
            animation_frames_list = self.create_correction_animation_frames(
                keypoints_3d, ideal_keypoints_3d, None, num_frames=animation_frames
            )
            
            # Check if we got any frames
            if not animation_frames_list:
                print("Warning: No animation frames generated, skipping animation for this frame")
                continue
            
            # Write frames to output video
            for frame_img in animation_frames_list:
                # Ensure frame has correct dimensions
                if frame_img.shape[:2] != (height, width):
                    frame_img = cv2.resize(frame_img, (width, height))
                out.write(frame_img)
            
            # Print progress
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Animation video complete. Output saved to {output_video_path}")
        return True
    
 
