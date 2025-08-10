import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from src.utils.keypoints_utils import normalize_keypoints, validate_keypoints


class Visualizer:
    def __init__(self, output_path='output/'):
        """
        Initialize the visualizer.
        
        Args:
            output_path: Path to save output files
        """
        self.output_path = output_path
        
        # MediaPipe pose connections for drawing
        self.pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head
            (0, 4), (4, 5), (5, 6), (6, 8),  # Head
            (9, 10),  # Mouth
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Waist
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
            (27, 29), (29, 31),  # Left foot
            (28, 30), (30, 32),  # Right foot
            (27, 31), (28, 32)   # Feet
        ]
        
    def create_3d_plot(self, keypoints_3d, feedback=None, title="3D Pose Visualization"):
        """
        Create a 3D plot of the pose.
        
        Args:
            keypoints_3d: 3D keypoints of the pose
            feedback: Dictionary containing feedback for each joint
            title: Title of the plot
            
        Returns:
            fig: matplotlib figure object
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if keypoints_3d is None:
            ax.text2D(0.5, 0.5, "No pose detected", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return fig
        
        # Normalize keypoints to numpy array
        keypoints_array = normalize_keypoints(keypoints_3d)
        
        if keypoints_array is None:
            ax.text2D(0.5, 0.5, "Invalid keypoints data", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return fig
        
        # Validate keypoints structure
        if not validate_keypoints(keypoints_array) or keypoints_array.shape[0] == 0:
            ax.text2D(0.5, 0.5, "Invalid keypoints format or empty array", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return fig
        
        # Extract x, y, z coordinates
        try:
            x = keypoints_array[:, 0]
            y = keypoints_array[:, 1]
            z = keypoints_array[:, 2]
        except IndexError:
            ax.text2D(0.5, 0.5, "Invalid keypoints dimensions", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return fig
        
        # Plot keypoints
        ax.scatter(x, y, z, c='blue', marker='o')
        
        # Draw connections
        for connection in self.pose_connections:
            if connection[0] < len(keypoints_array) and connection[1] < len(keypoints_array):
                ax.plot([x[connection[0]], x[connection[1]]],
                        [y[connection[0]], y[connection[1]]],
                        [z[connection[0]], z[connection[1]]], 'k-')
        
        # Color joints based on feedback
        if feedback:
            # Map joint names to keypoint indices
            joint_indices = {
                'right_elbow': 14,
                'right_knee': 26,
                'right_shoulder': 12,
                'right_hip': 24,
                'spine': 11  # Using shoulder as representative for spine
            }
            
            for joint, data in feedback.items():
                if joint in joint_indices and joint_indices[joint] < len(keypoints_array):
                    idx = joint_indices[joint]
                    color = 'green' if data['status'] == 'good' else 'red'
                    ax.scatter(x[idx], y[idx], z[idx], c=color, marker='o', s=100)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return fig



    
    def plot_to_image(self, fig):
        """
        Convert matplotlib figure to numpy array.
        
        Args:
            fig: matplotlib figure object
            
        Returns:
            img: numpy array representing the image
        """
        try:
            # Draw the figure
            fig.canvas.draw()
            
            # Get the RGBA buffer from the figure
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Close the figure to free memory
            plt.close(fig)
            
            return buf
        except Exception as e:
            print(f"Error converting plot to image: {e}")
            plt.close(fig)
            # Return a blank image if conversion fails
            return np.zeros((480, 640, 3), dtype=np.uint8)

    
    def create_side_by_side_comparison(self, keypoints_3d_current, keypoints_3d_ideal, 
                                      feedback_current, feedback_ideal=None,
                                      title="Pose Comparison"):
        """
        Create a side-by-side comparison of current and ideal poses.
        
        Args:
            keypoints_3d_current: 3D keypoints of the current pose
            keypoints_3d_ideal: 3D keypoints of the ideal pose
            feedback_current: Dictionary containing feedback for the current pose
            feedback_ideal: Dictionary containing feedback for the ideal pose
            title: Title of the plot
            
        Returns:
            fig: matplotlib figure object
        """
        # Create subplots
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot current pose
        self._plot_pose(ax1, keypoints_3d_current, feedback_current, "Current Pose")
        
        # Plot ideal pose
        self._plot_pose(ax2, keypoints_3d_ideal, feedback_ideal, "Ideal Pose")
        
        # Set overall title
        fig.suptitle(title)
        
        return fig

    
    def _plot_pose(self, ax, keypoints_3d, feedback, title):
        """
        Helper function to plot a single pose on the given axes.
        
        Args:
            ax: matplotlib 3D axes
            keypoints_3d: 3D keypoints of the pose
            feedback: Dictionary containing feedback for each joint
            title: Title of the subplot
        """
        if keypoints_3d is None:
            ax.text2D(0.5, 0.5, "No pose detected", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return
        
        # Convert keypoints to numpy array if it's not already
        if not isinstance(keypoints_3d, np.ndarray):
            try:
                # Handle different data structures
                if isinstance(keypoints_3d, dict):
                    # If it's a dictionary, extract values and convert to array
                    keypoints_array = np.array(list(keypoints_3d.values()))
                else:
                    # If it's a list or other iterable, convert directly
                    keypoints_array = np.array(keypoints_3d)
            except Exception as e:
                ax.text2D(0.5, 0.5, f"Error processing keypoints: {e}", transform=ax.transAxes, 
                        ha='center', va='center', fontsize=12)
                ax.set_title(title)
                return
        else:
            keypoints_array = keypoints_3d
        
        # Check if we have a 2D array with at least 2 columns
        if len(keypoints_array.shape) < 2 or keypoints_array.shape[1] < 3 or keypoints_array.shape[0] == 0:
            ax.text2D(0.5, 0.5, "Invalid keypoints format or empty array", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return
        
        # Extract x, y, z coordinates
        try:
            x = keypoints_array[:, 0]
            y = keypoints_array[:, 1]
            z = keypoints_array[:, 2]
        except IndexError:
            ax.text2D(0.5, 0.5, "Invalid keypoints dimensions", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return
        
        # Plot keypoints
        ax.scatter(x, y, z, c='blue', marker='o')
        
        # Draw connections
        for connection in self.pose_connections:
            if connection[0] < len(keypoints_array) and connection[1] < len(keypoints_array):
                ax.plot([x[connection[0]], x[connection[1]]],
                        [y[connection[0]], y[connection[1]]],
                        [z[connection[0]], z[connection[1]]], 'k-')
        
        # Color joints based on feedback
        if feedback:
            # Map joint names to keypoint indices
            joint_indices = {
                'right_elbow': 14,
                'right_knee': 26,
                'right_shoulder': 12,
                'right_hip': 24,
                'spine': 11  # Using shoulder as representative for spine
            }
            
            for joint, data in feedback.items():
                if joint in joint_indices and joint_indices[joint] < len(keypoints_array):
                    idx = joint_indices[joint]
                    color = 'green' if data['status'] == 'good' else 'red'
                    ax.scatter(x[idx], y[idx], z[idx], c=color, marker='o', s=100)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
    def generate_text_feedback(self, feedback):
        """
        Generate human-readable text feedback from pose analysis.
        
        Args:
            feedback: Dictionary containing feedback for each joint
            
        Returns:
            text_feedback: List of human-readable feedback strings
        """
        if not feedback or not isinstance(feedback, dict):
            return ["No feedback available"]
            
        text_feedback = []
        
        try:
            for joint, data in feedback.items():
                if isinstance(data, dict):
                    status = data.get('status', 'unknown')
                    value = data.get('value', 0)
                    ideal = data.get('ideal', 0)
                    
                    joint_name = joint.replace('_', ' ').title()
                    
                    if status == 'good':
                        continue  # Skip good poses for cleaner output
                    elif value < ideal:
                        text_feedback.append(f"{joint_name} angle too small: {value:.1f}° (ideal: {ideal}°)")
                    else:
                        text_feedback.append(f"{joint_name} angle too large: {value:.1f}° (ideal: {ideal}°)")
            
            # If no specific issues found, add a general positive feedback
            if not text_feedback:
                text_feedback.append("Good technique overall! Keep practicing.")
                
        except Exception as e:
            print(f"Error generating text feedback: {e}")
            text_feedback.append("Could not generate specific feedback due to analysis errors")
        
        return text_feedback
    
    def create_correction_animation_frames(self, keypoints_3d_current, keypoints_3d_ideal, corrections, num_frames=10):
        """
        Create frames for an animation showing the transition from current to ideal pose.
        
        Args:
            keypoints_3d_current: 3D keypoints of the current pose
            keypoints_3d_ideal: 3D keypoints of the ideal pose
            corrections: Dictionary containing correction vectors for each joint
            num_frames: Number of frames to generate
            
        Returns:
            frames: List of numpy arrays representing the frames
        """
        frames = []
        
        # Check if we have valid keypoints
        if keypoints_3d_current is None:
            print("Warning: Current keypoints is None, cannot create animation frames")
            return frames
            
        if keypoints_3d_ideal is None:
            print("Warning: Ideal keypoints is None, creating animation without ideal pose")
            # If no ideal pose, just duplicate the current pose
            for i in range(num_frames):
                frames.append(keypoints_3d_current.copy())
            return frames
        
        # Convert keypoints to numpy arrays if they're dictionaries
        if isinstance(keypoints_3d_current, dict):
            # Extract values from dictionary and convert to numpy array
            try:
                keypoints_3d_current = np.array(list(keypoints_3d_current.values()))
            except Exception as e:
                print(f"Error converting current keypoints to array: {e}")
                return frames
        
        if isinstance(keypoints_3d_ideal, dict):
            # Extract values from dictionary and convert to numpy array
            try:
                keypoints_3d_ideal = np.array(list(keypoints_3d_ideal.values()))
            except Exception as e:
                print(f"Error converting ideal keypoints to array: {e}")
                return frames
        
        # Check if keypoints are numpy arrays with the right shape
        if not isinstance(keypoints_3d_current, np.ndarray) or not isinstance(keypoints_3d_ideal, np.ndarray):
            print("Error: Keypoints are not numpy arrays after conversion")
            return frames
        
        if len(keypoints_3d_current.shape) < 2 or len(keypoints_3d_ideal.shape) < 2:
            print("Error: Keypoints arrays don't have enough dimensions")
            return frames
        
        if keypoints_3d_current.shape[1] < 3 or keypoints_3d_ideal.shape[1] < 3:
            print("Error: Keypoints arrays don't have enough dimensions")
            return frames
            
        # Check if shapes are compatible for interpolation
        if keypoints_3d_current.shape != keypoints_3d_ideal.shape:
            print(f"Warning: Shape mismatch between current keypoints {keypoints_3d_current.shape} and ideal keypoints {keypoints_3d_ideal.shape}")
            # Resize the arrays to match the smaller one
            min_points = min(keypoints_3d_current.shape[0], keypoints_3d_ideal.shape[0])
            keypoints_3d_current = keypoints_3d_current[:min_points]
            keypoints_3d_ideal = keypoints_3d_ideal[:min_points]
        
        try:
            # Final check to ensure both arrays are valid and have the same shape
            if keypoints_3d_current is None or keypoints_3d_ideal is None:
                raise ValueError("One or both keypoints arrays are None")
                
            if keypoints_3d_current.shape != keypoints_3d_ideal.shape:
                print(f"Warning: Shape mismatch after conversion. Current: {keypoints_3d_current.shape}, Ideal: {keypoints_3d_ideal.shape}")
                # Try to resize again to ensure compatibility
                min_points = min(keypoints_3d_current.shape[0], keypoints_3d_ideal.shape[0])
                keypoints_3d_current = keypoints_3d_current[:min_points]
                keypoints_3d_ideal = keypoints_3d_ideal[:min_points]
                
                if keypoints_3d_current.shape != keypoints_3d_ideal.shape:
                    raise ValueError(f"Cannot reconcile shape difference: {keypoints_3d_current.shape} vs {keypoints_3d_ideal.shape}")
            
            for i in range(num_frames):
                # Calculate interpolation factor
                t = i / (num_frames - 1)
                
                # Interpolate between current and ideal pose
                keypoints_3d_interpolated = keypoints_3d_current * (1 - t) + keypoints_3d_ideal * t
                
                # Create visualization
                fig = self.create_3d_plot(
                    keypoints_3d_interpolated,
                    feedback=None,
                    title=f"Pose Correction (Frame {i+1}/{num_frames})"
                )
                
                # Convert to image
                frame = self.plot_to_image(fig)
                frames.append(frame)
                
        except Exception as e:
            print(f"Error creating animation frames: {e}")
            # If interpolation fails, just use the current pose
            for i in range(num_frames):
                fig = self.create_3d_plot(
                    keypoints_3d_current,
                    feedback=None,
                    title=f"Pose Correction (Frame {i+1}/{num_frames})"
                )
                
                # Convert to image
                frame = self.plot_to_image(fig)
                frames.append(frame)
        
        return frames




