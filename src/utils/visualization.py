import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import plotly.graph_objects as go
from typing import List, Dict, Optional

class CricketVisualizer:
    def __init__(self, output_dir: str = "outputs"):

        """
        Initialize the CricketVisualizer class.
        
        Args:
            output_dir (str): Directory path to save outputs. Default is "outputs".
        """
        self.output_dir = Path(output_dir)  # Convert to Path object for easier path manipulation
        self.output_dir.mkdir(exist_ok=True)
        
    def create_3d_visualization_video(self, poses: List[np.ndarray], video_path: str) -> str:
        """Create 3D pose visualization video"""
        video_name = Path(video_path).stem
        output_path = self.output_dir / f"3d_visualization_{video_name}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (800, 600))
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for pose in poses:
            ax.clear()
            
            # Plot pose
            self._plot_pose_3d(ax, pose)
            
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(img)
            
        out.release()
        plt.close()
        
        return str(output_path)
    
    def _plot_pose_3d(self, ax: Axes3D, pose: np.ndarray):
        """Plot single pose in 3D"""
        # Define connections between keypoints
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head to shoulder
            (0, 4), (4, 5), (5, 6), (6, 8),  # Head to other shoulder
            (9, 10), (11, 12),  # Arms
            (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),  # Left arm
            (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),  # Right arm
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32)  # Right leg
        ]
        
        # Plot connections
        for connection in connections:
            points = pose[connection]
            ax.plot3D(*points.T, 'b-', linewidth=2)
        
        # Plot keypoints
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='r', s=50)
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
    
    def create_correction_overlay(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """Create correction overlay on frame"""
        # Add score overlay
        score_text = f"Score: {analysis.get('overall_score', 0):.2f}"
        cv2.putText(frame, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add feedback highlights
        feedback = analysis.get('feedback', [])
        y_offset = 70
        for i, feedback_item in enumerate(feedback[:3]):
            cv2.putText(frame, f"- {feedback_item}", (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def create_3d_skeleton(self, landmarks, title="3D Pose Analysis"):
        """
        Create a 3D skeleton visualization.
        
        Args:
            landmarks: 3D landmarks of the pose
            title: Title of the visualization
            
        Returns:
            fig: Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            
            if landmarks is None:
                fig = go.Figure()
                fig.add_annotation(text="No pose detected", 
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Extract x, y, z coordinates
            x = landmarks[:, 0]
            y = landmarks[:, 1]
            z = landmarks[:, 2]
            
            # Create 3D scatter plot for keypoints
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=z,  # Color by z-coordinate
                    colorscale='Viridis',
                    showscale=True
                )
            )])
            
            # Add connections between keypoints
            # Define connections (simplified for example)
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
                if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                    fig.add_trace(go.Scatter3d(
                        x=[x[connection[0]], x[connection[1]]],
                        y=[y[connection[0]], y[connection[1]]],
                        z=[z[connection[0]], z[connection[1]]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ))
            
            # Set layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )
            
            return fig
        except Exception as e:
            print(f"Error creating 3D skeleton: {e}")
            return None
