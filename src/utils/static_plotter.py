# src/utils/static_plotter.py
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Union

def plot_swing_trajectory(wrist_positions: List[Dict], output_path: str) -> None:
    """
    Plot the trajectory of the bat swing.
    
    Args:
        wrist_positions: List of dictionaries with 'x' and 'y' coordinates of wrist positions
        output_path: Path to save the plot
    """
    try:
        if not wrist_positions:
            print("No wrist positions provided for trajectory plot")
            return
        
        # Extract x and y coordinates
        x_coords = []
        y_coords = []
        
        for pos in wrist_positions:
            # Handle different input formats
            if isinstance(pos, dict):
                if 'x' in pos and 'y' in pos:
                    x_coords.append(pos['x'])
                    y_coords.append(pos['y'])
            elif isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                x_coords.append(pos[0])
                y_coords.append(pos[1])
        
        if not x_coords or not y_coords:
            print("No valid wrist coordinates found for trajectory plot")
            return
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot trajectory
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Bat Trajectory')
        
        # Mark start and end points
        plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End', zorder=5)
        
        # Add arrows to show direction
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            plt.arrow(x_coords[i-1], y_coords[i-1], dx, dy, 
                     head_width=0.02, head_length=0.03, fc='blue', ec='blue', alpha=0.5)
        
        # Set labels and title
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Bat Swing Trajectory Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Swing trajectory plot saved to {output_path}")
        
    except Exception as e:
        print(f"Error plotting swing trajectory: {str(e)}")
