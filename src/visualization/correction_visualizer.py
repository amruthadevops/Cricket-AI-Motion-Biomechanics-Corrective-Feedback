# src/visualization/correction_visualizer.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from .visualization import Visualizer
from src.utils.keypoints_utils import normalize_keypoints, validate_keypoints

class CorrectionVisualizer(Visualizer):
    def __init__(self, output_path='output/'):
        """
        Initialize the correction visualizer.
        
        Args:
            output_path: Path to save output files
        """
        super().__init__(output_path)
        
    def create_corrected_pose_visualization(self, keypoints_3d_current, keypoints_3d_ideal,
                                          feedback, corrections, title="Pose Correction"):
        """
        Create a visualization showing the current pose, ideal pose, and corrections.
        
        Args:
            keypoints_3d_current: 3D keypoints of the current pose
            keypoints_3d_ideal: 3D keypoints of the ideal pose
            feedback: Dictionary containing feedback for each joint
            corrections: Dictionary containing correction vectors for each joint
            title: Title of the plot
            
        Returns:
            fig: matplotlib figure object
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Current pose subplot
        ax1 = fig.add_subplot(131, projection='3d')
        self._plot_pose(ax1, keypoints_3d_current, feedback, "Current Pose")
        
        # Ideal pose subplot
        ax2 = fig.add_subplot(132, projection='3d')
        self._plot_pose(ax2, keypoints_3d_ideal, None, "Ideal Pose")
        
        # Corrections subplot
        ax3 = fig.add_subplot(133, projection='3d')
        self._plot_corrections(ax3, keypoints_3d_current, corrections, "Corrections")
        
        fig.suptitle(title)
        
        return fig
    
    def _convert_keypoints_to_array(self, keypoints_3d):
        """
        Convert keypoints to numpy array regardless of input format.
        
        Args:
            keypoints_3d: Keypoints in various formats (dict, list, array)
            
        Returns:
            numpy array or None if conversion fails
        """
        if keypoints_3d is None:
            return None
            
        try:
            # If already numpy array, validate and return
            if isinstance(keypoints_3d, np.ndarray):
                if len(keypoints_3d.shape) == 2 and keypoints_3d.shape[1] >= 3:
                    # Check if array is empty
                    if keypoints_3d.shape[0] == 0:
                        print("Warning: Empty keypoints array")
                        return None
                    return keypoints_3d
                elif len(keypoints_3d.shape) == 1:
                    # Reshape if it's a flattened array
                    if len(keypoints_3d) == 0 or len(keypoints_3d) % 3 != 0:
                        print("Warning: Invalid flattened keypoints array")
                        return None
                    return keypoints_3d.reshape(-1, 3)
                    
            # If it's a dictionary, extract values and convert to array
            elif isinstance(keypoints_3d, dict):
                keypoints_list = []
                for key, value in keypoints_3d.items():
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
                        keypoints_list.append(value[:3])  # Take only x, y, z
                if not keypoints_list:
                    print("Warning: No valid keypoints found in dictionary")
                    return None
                return np.array(keypoints_list)
                    
            # If it's a list or tuple
            elif isinstance(keypoints_3d, (list, tuple)):
                if not keypoints_3d:
                    print("Warning: Empty keypoints list")
                    return None
                    
                keypoints_array = np.array(keypoints_3d)
                if len(keypoints_array.shape) == 2 and keypoints_array.shape[1] >= 3:
                    return keypoints_array
                elif len(keypoints_array.shape) == 1 and len(keypoints_array) % 3 == 0:
                    return keypoints_array.reshape(-1, 3)
                else:
                    print(f"Warning: Invalid keypoints list shape: {keypoints_array.shape}")
                    return None
                    
        except Exception as e:
            print(f"Error converting keypoints to array: {e}")
            return None
            
        print("Warning: Unsupported keypoints format")
        return None
    
    def _plot_corrections(self, ax, keypoints_3d, corrections, title):
        """
        Helper function to plot correction vectors on the given axes.
        
        Args:
            ax: matplotlib 3D axes
            keypoints_3d: 3D keypoints of the pose
            corrections: Dictionary containing correction vectors for each joint
            title: Title of the subplot
        """
        keypoints_array = self._convert_keypoints_to_array(keypoints_3d)
        
        if keypoints_array is None or (isinstance(keypoints_array, np.ndarray) and (keypoints_array.shape[0] == 0 or len(keypoints_array.shape) < 2)):
            ax.text2D(0.5, 0.5, "No pose detected or invalid keypoints format", transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return
        
        # Extract x, y, z coordinates
        x = keypoints_array[:, 0]
        y = keypoints_array[:, 1]
        z = keypoints_array[:, 2]
        
        # Plot keypoints
        ax.scatter(x, y, z, c='blue', marker='o')
        
        # Draw connections
        connections = [
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
        ]
        
        for connection in connections:
            if connection[0] < len(keypoints_array) and connection[1] < len(keypoints_array):
                ax.plot([x[connection[0]], x[connection[1]]],
                        [y[connection[0]], y[connection[1]]],
                        [z[connection[0]], z[connection[1]]], 'k-')
        
        # Draw correction vectors
        if corrections:
            # Map joint names to keypoint indices
            joint_indices = {
                'right_elbow': 14,
                'right_knee': 26,
                'right_shoulder': 12,
                'right_hip': 24,
                'spine': 11  # Using shoulder as representative for spine
            }
            
            for joint, vector in corrections.items():
                if joint in joint_indices and joint_indices[joint] < len(keypoints_array):
                    idx = joint_indices[joint]
                    # Scale the vector for better visualization
                    scale = 0.2
                    if isinstance(vector, (list, tuple, np.ndarray)) and len(vector) >= 3:
                        ax.quiver(x[idx], y[idx], z[idx], 
                                vector[0]*scale, vector[1]*scale, vector[2]*scale,
                                color='red', arrow_length_ratio=0.1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        if len(x) > 0:
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

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
        
        # Convert keypoints to numpy arrays
        keypoints_current = self._convert_keypoints_to_array(keypoints_3d_current)
        keypoints_ideal = self._convert_keypoints_to_array(keypoints_3d_ideal)
        
        # Check if shapes are compatible for interpolation
        if keypoints_current is not None and keypoints_ideal is not None:
            if keypoints_current.shape != keypoints_ideal.shape:
                print(f"Warning: Shape mismatch between current keypoints {keypoints_current.shape} and ideal keypoints {keypoints_ideal.shape}")
                # Resize the arrays to match the smaller one
                min_points = min(keypoints_current.shape[0], keypoints_ideal.shape[0])
                keypoints_current = keypoints_current[:min_points]
                keypoints_ideal = keypoints_ideal[:min_points]
        
        # Check if we have valid keypoints
        if keypoints_current is None:
            print("Warning: Current keypoints is None, cannot create animation frames")
            return frames
            
        if keypoints_ideal is None:
            print("Warning: Ideal keypoints is None, creating animation without ideal pose")
            # If no ideal pose, just duplicate the current pose
            for i in range(num_frames):
                try:
                    fig = self.create_3d_plot(
                        keypoints_current,
                        feedback=None,
                        title=f"Pose Correction (Frame {i+1}/{num_frames})"
                    )
                    frame = self.plot_to_image(fig)
                    frames.append(frame)
                except Exception as e:
                    print(f"Error creating frame {i}: {e}")
                    break
            return frames
        
        try:
            # Final check to ensure both arrays are valid and have the same shape
            if keypoints_current is None or keypoints_ideal is None:
                raise ValueError("One or both keypoints arrays are None")
                
            if keypoints_current.shape != keypoints_ideal.shape:
                print(f"Warning: Shape mismatch after conversion. Current: {keypoints_current.shape}, Ideal: {keypoints_ideal.shape}")
                # Try to resize again to ensure compatibility
                min_points = min(keypoints_current.shape[0], keypoints_ideal.shape[0])
                keypoints_current = keypoints_current[:min_points]
                keypoints_ideal = keypoints_ideal[:min_points]
                
                if keypoints_current.shape != keypoints_ideal.shape:
                    raise ValueError(f"Cannot reconcile shape difference: {keypoints_current.shape} vs {keypoints_ideal.shape}")
            
            for i in range(num_frames):
                # Calculate interpolation factor
                t = i / (num_frames - 1) if num_frames > 1 else 0
                
                # Interpolate between current and ideal pose (now both are numpy arrays)
                keypoints_3d_interpolated = keypoints_current * (1 - t) + keypoints_ideal * t
                
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
                try:
                    fig = self.create_3d_plot(
                        keypoints_current,
                        feedback=None,
                        title=f"Pose Correction (Frame {i+1}/{num_frames})"
                    )
                    
                    # Convert to image
                    frame = self.plot_to_image(fig)
                    frames.append(frame)
                except Exception as inner_e:
                    print(f"Error creating fallback frame {i}: {inner_e}")
                    break
        
        return frames

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
            
            # Convert RGB to BGR for OpenCV
            buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
            # Close the figure to free memory
            plt.close(fig)
            
            return buf
        except Exception as e:
            print(f"Error converting plot to image: {e}")
            plt.close(fig)
            # Return a blank image if conversion fails
            return np.zeros((480, 640, 3), dtype=np.uint8)