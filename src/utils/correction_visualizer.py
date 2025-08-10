# src/utils/correction_visualizer.py
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime

from .json_utils import convert_numpy_to_json_serializable
from .visualization import CricketVisualizer
from .metrics import (
    calculate_angle, calculate_distance, calculate_arm_angle, 
    calculate_torque, estimate_shot_type, detect_follow_through_balance,
    calculate_center_of_gravity_shift, analyze_wrist_motion_path,
    estimate_shoulder_hip_torque_transfer
)

class CricketCorrectionVisualizer(CricketVisualizer):
    def __init__(self):
        super().__init__()
        self.correction_colors = {
            "high": (0, 0, 255),      # Red
            "medium": (0, 165, 255),  # Orange
            "low": (255, 255, 0)      # Yellow
        }
    
    def calculate_overall_score(self, comprehensive_results: Dict, primary_activity: str) -> float:
        """
        Calculate overall score from comprehensive results.
        
        Args:
            comprehensive_results: Dictionary with analysis results
            primary_activity: Type of cricket activity
            
        Returns:
            Overall score (0-1)
        """
        try:
            # Get scores based on activity type
            if primary_activity == "batting":
                mechanics = comprehensive_results.get("batting_mechanics", {})
                scores = mechanics.get("scores", {})
                return scores.get("overall_score", 0.0)
            
            elif primary_activity == "bowling":
                mechanics = comprehensive_results.get("bowling_mechanics", {})
                scores = mechanics.get("scores", {})
                return scores.get("overall_score", 0.0)
            
            elif primary_activity == "fielding":
                mechanics = comprehensive_results.get("fielding_mechanics", {})
                scores = mechanics.get("scores", {})
                return scores.get("overall_score", 0.0)
            
            elif primary_activity == "follow_through":
                mechanics = comprehensive_results.get("follow_through", {})
                scores = mechanics.get("scores", {})
                return scores.get("overall_score", 0.0)
            
            else:
                # Default calculation for unknown activity
                scores = comprehensive_results.get("scores", {})
                return scores.get("overall_score", 0.0)
                
        except Exception as e:
            print(f"Error calculating overall score: {str(e)}")
            return 0.0
    
    def identify_improvement_areas(self, comprehensive_results: Dict, primary_activity: str) -> List[str]:
        """
        Identify improvement areas from comprehensive results.
        
        Args:
            comprehensive_results: Dictionary with analysis results
            primary_activity: Type of cricket activity
            
        Returns:
            List of improvement areas
        """
        try:
            improvement_areas = []
            
            # Get corrections based on activity type
            if primary_activity == "batting":
                mechanics = comprehensive_results.get("batting_mechanics", {})
                corrections = mechanics.get("corrections", [])
                
                # Analyze specific areas
                stance = mechanics.get("stance", {})
                if stance.get("score", 0) < 0.7:
                    improvement_areas.append("Batting Stance")
                
                timing = mechanics.get("timing", {})
                if timing.get("score", 0) < 0.7:
                    improvement_areas.append("Batting Timing")
                
                bat_angle = mechanics.get("bat_angle", {})
                if bat_angle.get("score", 0) < 0.7:
                    improvement_areas.append("Bat Angle")
                
                foot_placement = mechanics.get("foot_placement", {})
                if foot_placement.get("score", 0) < 0.7:
                    improvement_areas.append("Foot Placement")
            
            elif primary_activity == "bowling":
                mechanics = comprehensive_results.get("bowling_mechanics", {})
                corrections = mechanics.get("corrections", [])
                
                # Analyze specific areas
                release_dynamics = mechanics.get("release_dynamics", {})
                if release_dynamics.get("score", 0) < 0.7:
                    improvement_areas.append("Release Dynamics")
                
                run_up_consistency = mechanics.get("run_up_consistency", {})
                if run_up_consistency.get("score", 0) < 0.7:
                    improvement_areas.append("Run-up Consistency")
                
                front_foot_landing = mechanics.get("front_foot_landing", {})
                if front_foot_landing.get("score", 0) < 0.7:
                    improvement_areas.append("Front Foot Landing")
            
            elif primary_activity == "fielding":
                mechanics = comprehensive_results.get("fielding_mechanics", {})
                corrections = mechanics.get("corrections", [])
                
                # Analyze specific areas
                dive_analysis = mechanics.get("dive_analysis", {})
                if dive_analysis.get("score", 0) < 0.7:
                    improvement_areas.append("Diving Technique")
                
                throw_mechanics = mechanics.get("throw_mechanics", {})
                if throw_mechanics.get("score", 0) < 0.7:
                    improvement_areas.append("Throwing Mechanics")
                
                anticipation_reaction = mechanics.get("anticipation_reaction", {})
                if anticipation_reaction.get("score", 0) < 0.7:
                    improvement_areas.append("Anticipation and Reaction")
            
            elif primary_activity == "follow_through":
                mechanics = comprehensive_results.get("follow_through", {})
                corrections = mechanics.get("corrections", [])
                
                # Analyze specific areas
                balance = mechanics.get("balance", {})
                if balance.get("score", 0) < 0.7:
                    improvement_areas.append("Balance")
                
                wrist_motion = mechanics.get("wrist_motion", {})
                if wrist_motion.get("score", 0) < 0.7:
                    improvement_areas.append("Wrist Motion")
                
                torque_transfer = mechanics.get("torque_transfer", {})
                if torque_transfer.get("score", 0) < 0.7:
                    improvement_areas.append("Torque Transfer")
            
            # If no specific areas identified, use general corrections
            if not improvement_areas and corrections:
                improvement_areas = corrections[:3]  # Take top 3 corrections
            
            # If still no areas, provide default
            if not improvement_areas:
                improvement_areas = ["General Technique Improvement"]
            
            return improvement_areas
            
        except Exception as e:
            print(f"Error identifying improvement areas: {str(e)}")
            return ["General Technique Improvement"]
    
    def identify_strengths(self, comprehensive_results: Dict, primary_activity: str) -> List[str]:
        """
        Identify strengths from comprehensive results.
        
        Args:
            comprehensive_results: Dictionary with analysis results
            primary_activity: Type of cricket activity
            
        Returns:
            List of strengths
        """
        try:
            strengths = []
            
            # Get analysis based on activity type
            if primary_activity == "batting":
                mechanics = comprehensive_results.get("batting_mechanics", {})

                # Analyze specific areas
                stance = mechanics.get("stance", {})
                if stance.get("score", 0) >= 0.8:
                    strengths.append("Batting Stance")
                
                timing = mechanics.get("timing", {})
                if timing.get("score", 0) >= 0.8:
                    strengths.append("Batting Timing")
                
                bat_angle = mechanics.get("bat_angle", {})
                if bat_angle.get("score", 0) >= 0.8:
                    strengths.append("Bat Angle Control")
                
                foot_placement = mechanics.get("foot_placement", {})
                if foot_placement.get("score", 0) >= 0.8:
                    strengths.append("Foot Placement")
            
            elif primary_activity == "bowling":
                mechanics = comprehensive_results.get("bowling_mechanics", {})
                
                # Analyze specific areas
                release_dynamics = mechanics.get("release_dynamics", {})
                if release_dynamics.get("score", 0) >= 0.8:
                    strengths.append("Release Dynamics")
                
                run_up_consistency = mechanics.get("run_up_consistency", {})
                if run_up_consistency.get("score", 0) >= 0.8:
                    strengths.append("Run-up Consistency")
                
                front_foot_landing = mechanics.get("front_foot_landing", {})
                if front_foot_landing.get("score", 0) >= 0.8:
                    strengths.append("Front Foot Landing")
            
            elif primary_activity == "fielding":
                mechanics = comprehensive_results.get("fielding_mechanics", {})
                
                # Analyze specific areas
                dive_analysis = mechanics.get("dive_analysis", {})
                if dive_analysis.get("score", 0) >= 0.8:
                    strengths.append("Diving Technique")
                
                throw_mechanics = mechanics.get("throw_mechanics", {})
                if throw_mechanics.get("score", 0) >= 0.8:
                    strengths.append("Throwing Mechanics")
                
                anticipation_reaction = mechanics.get("anticipation_reaction", {})
                if anticipation_reaction.get("score", 0) >= 0.8:
                    strengths.append("Anticipation and Reaction")
            
            elif primary_activity == "follow_through":
                mechanics = comprehensive_results.get("follow_through", {})
                
                # Analyze specific areas
                balance = mechanics.get("balance", {})
                if balance.get("score", 0) >= 0.8:
                    strengths.append("Balance")
                
                wrist_motion = mechanics.get("wrist_motion", {})
                if wrist_motion.get("score", 0) >= 0.8:
                    strengths.append("Wrist Motion")
                
                torque_transfer = mechanics.get("torque_transfer", {})
                if torque_transfer.get("score", 0) >= 0.8:
                    strengths.append("Torque Transfer")
            
            # If no specific strengths identified, provide default
            if not strengths:
                strengths = ["Good Overall Technique"]
            
            return strengths
            
        except Exception as e:
            print(f"Error identifying strengths: {str(e)}")
            return ["Good Overall Technique"]
    
    def generate_correction_recommendations(self, comprehensive_results: Dict, primary_activity: str) -> List[Dict]:
        """
        Generate detailed correction recommendations based on analysis results.
        
        Args:
            comprehensive_results: Dictionary with analysis results
            primary_activity: Type of cricket activity
            
        Returns:
            List of recommendation dictionaries with text, priority, and score
        """
        try:
            recommendations = []
            
            # Get analysis based on activity type
            if primary_activity == "batting":
                mechanics = comprehensive_results.get("batting_mechanics", {})
                corrections = mechanics.get("corrections", [])
                
                # Convert corrections to recommendation format
                for correction in corrections[:5]:  # Limit to top 5
                    # Determine priority based on correction content
                    if "improve" in correction.lower() or "poor" in correction.lower():
                        priority = "high"
                    elif "could" in correction.lower() or "work on" in correction.lower():
                        priority = "medium"
                    else:
                        priority = "low"
                    
                    # Estimate score (inverse of severity)
                    if priority == "high":
                        score = 0.3
                    elif priority == "medium":
                        score = 0.6
                    else:
                        score = 0.8
                    
                    recommendations.append({
                        "text": correction,
                        "priority": priority,
                        "score": score
                    })
            
            elif primary_activity == "bowling":
                mechanics = comprehensive_results.get("bowling_mechanics", {})
                corrections = mechanics.get("corrections", [])
                
                # Convert corrections to recommendation format
                for correction in corrections[:5]:  # Limit to top 5
                    # Determine priority based on correction content
                    if "improve" in correction.lower() or "poor" in correction.lower():
                        priority = "high"
                    elif "could" in correction.lower() or "work on" in correction.lower():
                        priority = "medium"
                    else:
                        priority = "low"
                    
                    # Estimate score (inverse of severity)
                    if priority == "high":
                        score = 0.3
                    elif priority == "medium":
                        score = 0.6
                    else:
                        score = 0.8
                    
                    recommendations.append({
                        "text": correction,
                        "priority": priority,
                        "score": score
                    })
            
            elif primary_activity == "fielding":
                mechanics = comprehensive_results.get("fielding_mechanics", {})
                corrections = mechanics.get("corrections", [])
                
                # Convert corrections to recommendation format
                for correction in corrections[:5]:  # Limit to top 5
                    # Determine priority based on correction content
                    if "improve" in correction.lower() or "poor" in correction.lower():
                        priority = "high"
                    elif "could" in correction.lower() or "work on" in correction.lower():
                        priority = "medium"
                    else:
                        priority = "low"
                    
                    # Estimate score (inverse of severity)
                    if priority == "high":
                        score = 0.3
                    elif priority == "medium":
                        score = 0.6
                    else:
                        score = 0.8
                    
                    recommendations.append({
                        "text": correction,
                        "priority": priority,
                        "score": score
                    })
            
            elif primary_activity == "follow_through":
                mechanics = comprehensive_results.get("follow_through", {})
                corrections = mechanics.get("corrections", [])
                
                # Convert corrections to recommendation format
                for correction in corrections[:5]:  # Limit to top 5
                    # Determine priority based on correction content
                    if "improve" in correction.lower() or "poor" in correction.lower():
                        priority = "high"
                    elif "could" in correction.lower() or "work on" in correction.lower():
                        priority = "medium"
                    else:
                        priority = "low"
                    
                    # Estimate score (inverse of severity)
                    if priority == "high":
                        score = 0.3
                    elif priority == "medium":
                        score = 0.6
                    else:
                        score = 0.8
                    
                    recommendations.append({
                        "text": correction,
                        "priority": priority,
                        "score": score
                    })
            
            # If no specific recommendations identified, provide default
            if not recommendations:
                recommendations = [
                    {
                        "text": "Continue practicing to improve overall technique",
                        "priority": "medium",
                        "score": 0.7
                    }
                ]
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating correction recommendations: {str(e)}")
            return [
                {
                    "text": "Error generating recommendations",
                    "priority": "high",
                    "score": 0.0
                }
            ]
    
    def visualize_corrections_on_frame(self, frame: np.ndarray, corrections: List[str], 
                                    scores: List[float], activity_type: str = "unknown") -> np.ndarray:
        """
        Visualize corrective feedback on a frame with scores.
        
        Args:
            frame: Input frame as numpy array
            corrections: List of correction strings
            scores: List of scores corresponding to corrections
            activity_type: Type of cricket activity
            
        Returns:
            Frame with visualized corrections
        """
        try:
            # Make a copy of the frame to avoid modifying the original
            output_frame = frame.copy()
            
            # Get color based on activity type
            base_color = self.colors.get(activity_type, (255, 255, 255))  # Default to white
            
            # Add corrections text with scores
            y_offset = 30
            for i, (correction, score) in enumerate(zip(corrections, scores)):
                # Determine color based on score
                if score < 0.4:
                    color = self.correction_colors["high"]  # Red for low scores
                elif score < 0.7:
                    color = self.correction_colors["medium"]  # Orange for medium scores
                else:
                    color = self.correction_colors["low"]  # Yellow for high scores
                
                # Add correction text
                cv2.putText(
                    output_frame, 
                    f"{correction} ({score:.2f})", 
                    (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    color, 
                    2
                )
                y_offset += 30
            
            # Add overall score indicator
            if scores:
                overall_score = np.mean(scores)
                score_text = f"Overall Score: {overall_score:.2f}"
                score_color = base_color
                
                cv2.putText(
                    output_frame, 
                    score_text, 
                    (10, y_offset + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    score_color, 
                    2
                )
            
            return output_frame
            
        except Exception as e:
            print(f"Error visualizing corrections on frame: {str(e)}")
            return frame
    
    def create_correction_video(self, frames_data: List[Dict], comprehensive_results: Dict, 
                              primary_activity: str, output_path: str) -> None:
        """
        Create a video with visualized corrections.
        
        Args:
            frames_data: List of frame data
            comprehensive_results: Dictionary with analysis results
            primary_activity: Type of cricket activity
            output_path: Path to save the video
        """
        try:
            if not frames_data:
                print("No frame data provided for correction video")
                return
            
            # Get corrections and scores
            corrections = comprehensive_results.get("corrections", [])
            scores = []
            
            # Estimate scores based on activity type
            if primary_activity == "batting":
                mechanics = comprehensive_results.get("batting_mechanics", {})
                scores = [
                    mechanics.get("stance", {}).get("score", 0.5),
                    mechanics.get("timing", {}).get("score", 0.5),
                    mechanics.get("bat_angle", {}).get("score", 0.5),
                    mechanics.get("foot_placement", {}).get("score", 0.5)
                ]
            elif primary_activity == "bowling":
                mechanics = comprehensive_results.get("bowling_mechanics", {})
                scores = [
                    mechanics.get("release_dynamics", {}).get("score", 0.5),
                    mechanics.get("run_up_consistency", {}).get("score", 0.5),
                    mechanics.get("front_foot_landing", {}).get("score", 0.5)
                ]
            elif primary_activity == "fielding":
                mechanics = comprehensive_results.get("fielding_mechanics", {})
                scores = [
                    mechanics.get("dive_analysis", {}).get("score", 0.5),
                    mechanics.get("throw_mechanics", {}).get("score", 0.5),
                    mechanics.get("anticipation_reaction", {}).get("score", 0.5)
                ]
            elif primary_activity == "follow_through":
                mechanics = comprehensive_results.get("follow_through", {})
                scores = [
                    mechanics.get("balance", {}).get("score", 0.5),
                    mechanics.get("wrist_motion", {}).get("score", 0.5),
                    mechanics.get("torque_transfer", {}).get("score", 0.5)
                ]
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get frame dimensions
            sample_frame = frames_data[0].get("frame")
            if sample_frame is None:
                print("No frame image found in frame data")
                return
            
            height, width = sample_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            # Process each frame
            for frame_data in frames_data:
                frame = frame_data.get("frame")
                if frame is None:
                    continue
                
                # Visualize corrections on frame
                corrected_frame = self.visualize_corrections_on_frame(
                    frame, corrections, scores, primary_activity
                )
                
                # Write frame to video
                out.write(corrected_frame)
            
            # Release video writer
            out.release()
            
            print(f"Correction video saved to {output_path}")
            
        except Exception as e:
            print(f"Error creating correction video: {str(e)}")
    
    def generate_correction_report(self, comprehensive_results: Dict, primary_activity: str, 
                                output_dir: str = "outputs") -> str:
        """
        Generate a comprehensive correction report.
        
        Args:
            comprehensive_results: Dictionary with analysis results
            primary_activity: Type of cricket activity
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved report
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculate overall score
            overall_score = self.calculate_overall_score(comprehensive_results, primary_activity)
            
            # Identify improvement areas
            improvement_areas = self.identify_improvement_areas(comprehensive_results, primary_activity)
            
            # Identify strengths
            strengths = self.identify_strengths(comprehensive_results, primary_activity)
            
            # Generate correction recommendations
            recommendations = self.generate_correction_recommendations(comprehensive_results, primary_activity)
            
            # Prepare report data
            report_data = {
                "timestamp": str(datetime.now()),
                "primary_activity": primary_activity,
                "overall_score": overall_score,
                "improvement_areas": improvement_areas,
                "strengths": strengths,
                "recommendations": recommendations,
                "detailed_analysis": comprehensive_results
            }
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correction_report_{primary_activity}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save report to file
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating correction report: {str(e)}")
            return ""


    def generate_batting_report(self, comprehensive_results: Dict, output_dir: str = "outputs") -> str:
        """
        Generate a comprehensive batting analysis report.
        
        Args:
            comprehensive_results: Dictionary with analysis results
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved report
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get batting mechanics
            batting_mechanics = comprehensive_results.get("batting_mechanics", {})
            
            # Prepare report data
            report_data = {
                "timestamp": str(datetime.now()),
                "report_type": "batting_analysis",
                "summary": {
                    "overall_score": batting_mechanics.get("scores", {}).get("overall_score", 0.0),
                    "shot_type": batting_mechanics.get("shot_type", {}).get("type", "unknown"),
                    "trigger_movement": batting_mechanics.get("trigger_movement", {}).get("type", "unknown")
                },
                "detailed_analysis": {
                    "stance": batting_mechanics.get("stance", {}),
                    "timing": batting_mechanics.get("timing", {}),
                    "bat_angle": batting_mechanics.get("bat_angle", {}),
                    "foot_placement": batting_mechanics.get("foot_placement", {})
                },
                "trajectory": batting_mechanics.get("trajectory", []),
                "corrections": batting_mechanics.get("corrections", []),
                "recommendations": self.generate_correction_recommendations(
                    comprehensive_results, "batting"
                )
            }
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batting_analysis_report_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save report to file
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating batting report: {str(e)}")
            return ""

    def create_pose_comparison_3d(self, current_landmarks, activity_type, analysis_results):
        """
        Create a 3D pose comparison visualization.
        
        Args:
            current_landmarks: Current pose landmarks
            activity_type: Type of activity (batting, bowling, fielding)
            analysis_results: Analysis results
            
        Returns:
            fig: Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Get ideal landmarks
            ideal_landmarks = self.get_ideal_keypoints(activity_type)
            
            if ideal_landmarks is None:
                # If no ideal landmarks, just show current pose
                return self.create_3d_skeleton(current_landmarks, f"Current {activity_type.title()} Pose")
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=['Current Pose', 'Ideal Pose']
            )
            
            # Add current pose
            x_current = current_landmarks[:, 0]
            y_current = current_landmarks[:, 1]
            z_current = current_landmarks[:, 2]
            
            fig.add_trace(
                go.Scatter3d(
                    x=x_current, y=y_current, z=z_current,
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name='Current Pose'
                ),
                row=1, col=1
            )
            
            # Add ideal pose
            x_ideal = ideal_landmarks[:, 0]
            y_ideal = ideal_landmarks[:, 1]
            z_ideal = ideal_landmarks[:, 2]
            
            fig.add_trace(
                go.Scatter3d(
                    x=x_ideal, y=y_ideal, z=z_ideal,
                    mode='markers',
                    marker=dict(size=5, color='green'),
                    name='Ideal Pose'
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"{activity_type.title()} Pose Comparison",
                height=600
            )
            
            return fig
        except Exception as e:
            print(f"Error creating pose comparison: {e}")
            return None

    def create_correction_dashboard(self, analysis_results, frames_data, activity_type):
        """
        Create a comprehensive correction dashboard.
        
        Args:
            analysis_results: Analysis results
            frames_data: Frame data
            activity_type: Type of activity
            
        Returns:
            fig: Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{'type': 'bar'}, {'type': 'pie'}],
                    [{'type': 'scatter'}, {'type': 'table'}]],
                subplot_titles=['Scores by Category', 'Activity Distribution', 
                            'Performance Over Time', 'Recommendations']
            )
            
            # Extract scores
            categories = []
            scores = []
            
            for mechanic, results in analysis_results.items():
                if isinstance(results, dict) and 'scores' in results and 'overall_score' in results['scores']:
                    categories.append(mechanic.replace('_', ' ').title())
                    scores.append(results['scores']['overall_score'])
            
            # Add bar chart
            fig.add_trace(
                go.Bar(x=categories, y=scores, name='Scores'),
                row=1, col=1
            )
            
            # Add pie chart for activity distribution (placeholder)
            fig.add_trace(
                go.Pie(labels=['Batting', 'Bowling', 'Fielding'], 
                    values=[0.4, 0.3, 0.3], name='Activity'),
                row=1, col=2
            )
            
            # Add scatter plot for performance over time (placeholder)
            fig.add_trace(
                go.Scatter(x=list(range(len(frames_data))), 
                        y=[0.7] * len(frames_data), 
                        mode='lines', name='Performance'),
                row=2, col=1
            )
            
            # Add table for recommendations (placeholder)
            fig.add_trace(
                go.Table(
                    header=dict(values=['Priority', 'Recommendation']),
                    cells=dict(values=[
                        ['High', 'Medium', 'Low'],
                        ['Improve stance', 'Work on timing', 'Practice follow-through']
                    ])
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"{activity_type.title()} Analysis Dashboard",
                height=800
            )
            
            return fig
        except Exception as e:
            print(f"Error creating correction dashboard: {e}")
            return None

    def save_correction_analysis(self, analysis_results, activity_type, video_path, output_dir="outputs"):
        """
        Save correction analysis to file.
        
        Args:
            analysis_results: Analysis results
            activity_type: Type of activity
            video_path: Path to the video file
            output_dir: Output directory
            
        Returns:
            str: Path to the saved file
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f"correction_analysis_{video_name}_{timestamp}.json")
            
            # Prepare data
            data = {
                "video_path": video_path,
                "activity_type": activity_type,
                "timestamp": timestamp,
                "analysis_results": analysis_results,
                "recommendations": self.generate_correction_recommendations(analysis_results, activity_type),
                "overall_score": self.calculate_overall_score(analysis_results, activity_type),
                "improvement_areas": self.identify_improvement_areas(analysis_results, activity_type),
                "strengths": self.identify_strengths(analysis_results, activity_type)
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(convert_numpy_to_json_serializable(data), f, indent=2)
            
            return output_path
        except Exception as e:
            print(f"Error saving correction analysis: {e}")
            return None

    def get_ideal_keypoints(self, activity_type):
        """
        Get ideal keypoints for the specified activity type.
        
        Args:
            activity_type (str): Type of activity (batting, bowling, fielding)
            
        Returns:
            numpy.ndarray: Ideal keypoints for the activity, or None if not available
        """
        ideal_poses_dir = "models/ideal_poses"
        
        # Map activity types to ideal pose files
        activity_to_file = {
            "batting": "ideal_batting_pose.json",
            "bowling": "ideal_bowling_pose.json",
            "fielding": "ideal_fielding_pose.json"
        }
        
        if activity_type not in activity_to_file:
            return None
            
        ideal_pose_file = os.path.join(ideal_poses_dir, activity_to_file[activity_type])
        
        if not os.path.exists(ideal_pose_file):
            print(f"⚠️ Ideal pose file not found: {ideal_pose_file}")
            return None
            
        try:
            with open(ideal_pose_file, 'r') as f:
                ideal_pose_data = json.load(f)
                # Return as dictionary, not numpy array
                return ideal_pose_data['keypoints_3d']
        except Exception as e:
            print(f"⚠️ Error loading ideal pose: {e}")
            return None

        
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
        
        try:
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
            
        
    def analyze_pose(self, keypoints_3d):
        """
        Analyze a 3D pose and generate feedback.
        
        Args:
            keypoints_3d: 3D keypoints of the pose
            
        Returns:
            tuple: (feedback, score)
                feedback: Dictionary containing feedback for each joint
                score: Overall score of the pose
        """
        if keypoints_3d is None:
            return {}, 0.0
            
        try:
            # Initialize feedback and score
            feedback = {}
            scores = []
            
            # Define key angles and their ideal values
            key_angles = {
                'elbow_angle': {'ideal': 90, 'tolerance': 15},
                'knee_angle': {'ideal': 170, 'tolerance': 10},
                'shoulder_angle': {'ideal': 80, 'tolerance': 15},
                'hip_angle': {'ideal': 170, 'tolerance': 10},
                'spine_angle': {'ideal': 10, 'tolerance': 5}
            }
            
            # Calculate angles if we have the necessary keypoints
            if 'right_shoulder' in keypoints_3d and 'right_elbow' in keypoints_3d and 'right_wrist' in keypoints_3d:
                # Calculate elbow angle
                elbow_angle = self.calculate_angle(
                    keypoints_3d['right_shoulder'],
                    keypoints_3d['right_elbow'],
                    keypoints_3d['right_wrist']
                )
                feedback['right_elbow'] = {
                    'value': elbow_angle,
                    'ideal': key_angles['elbow_angle']['ideal'],
                    'status': 'good' if abs(elbow_angle - key_angles['elbow_angle']['ideal']) <= key_angles['elbow_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_elbow']['status'] == 'good' else 0.5)
            
            if 'right_hip' in keypoints_3d and 'right_knee' in keypoints_3d and 'right_ankle' in keypoints_3d:
                # Calculate knee angle
                knee_angle = self.calculate_angle(
                    keypoints_3d['right_hip'],
                    keypoints_3d['right_knee'],
                    keypoints_3d['right_ankle']
                )
                feedback['right_knee'] = {
                    'value': knee_angle,
                    'ideal': key_angles['knee_angle']['ideal'],
                    'status': 'good' if abs(knee_angle - key_angles['knee_angle']['ideal']) <= key_angles['knee_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_knee']['status'] == 'good' else 0.5)
            
            if 'right_elbow' in keypoints_3d and 'right_shoulder' in keypoints_3d and 'right_hip' in keypoints_3d:
                # Calculate shoulder angle
                shoulder_angle = self.calculate_angle(
                    keypoints_3d['right_elbow'],
                    keypoints_3d['right_shoulder'],
                    keypoints_3d['right_hip']
                )
                feedback['right_shoulder'] = {
                    'value': shoulder_angle,
                    'ideal': key_angles['shoulder_angle']['ideal'],
                    'status': 'good' if abs(shoulder_angle - key_angles['shoulder_angle']['ideal']) <= key_angles['shoulder_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_shoulder']['status'] == 'good' else 0.5)
            
            if 'right_shoulder' in keypoints_3d and 'right_hip' in keypoints_3d and 'right_knee' in keypoints_3d:
                # Calculate hip angle
                hip_angle = self.calculate_angle(
                    keypoints_3d['right_shoulder'],
                    keypoints_3d['right_hip'],
                    keypoints_3d['right_knee']
                )
                feedback['right_hip'] = {
                    'value': hip_angle,
                    'ideal': key_angles['hip_angle']['ideal'],
                    'status': 'good' if abs(hip_angle - key_angles['hip_angle']['ideal']) <= key_angles['hip_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_hip']['status'] == 'good' else 0.5)
            
            # Calculate overall score
            overall_score = np.mean(scores) if scores else 0.0
            
            return feedback, overall_score
            
        except Exception as e:
            print(f"Error analyzing pose: {e}")
            return {}, 0.0


    
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
    
    def analyze_pose(self, keypoints_3d):
        """
        Analyze a 3D pose and generate feedback.
        
        Args:
            keypoints_3d: 3D keypoints of the pose
            
        Returns:
            tuple: (feedback, score)
                feedback: Dictionary containing feedback for each joint
                score: Overall score of the pose
        """
        if keypoints_3d is None:
            return {}, 0.0
            
        try:
            # Initialize feedback and score
            feedback = {}
            scores = []
            
            # Define key angles and their ideal values
            key_angles = {
                'elbow_angle': {'ideal': 90, 'tolerance': 15},
                'knee_angle': {'ideal': 170, 'tolerance': 10},
                'shoulder_angle': {'ideal': 80, 'tolerance': 15},
                'hip_angle': {'ideal': 170, 'tolerance': 10},
                'spine_angle': {'ideal': 10, 'tolerance': 5}
            }
            
            # Calculate angles if we have the necessary keypoints
            if 'right_shoulder' in keypoints_3d and 'right_elbow' in keypoints_3d and 'right_wrist' in keypoints_3d:
                # Calculate elbow angle
                elbow_angle = self.calculate_angle(
                    keypoints_3d['right_shoulder'],
                    keypoints_3d['right_elbow'],
                    keypoints_3d['right_wrist']
                )
                feedback['right_elbow'] = {
                    'value': elbow_angle,
                    'ideal': key_angles['elbow_angle']['ideal'],
                    'status': 'good' if abs(elbow_angle - key_angles['elbow_angle']['ideal']) <= key_angles['elbow_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_elbow']['status'] == 'good' else 0.5)
            
            if 'right_hip' in keypoints_3d and 'right_knee' in keypoints_3d and 'right_ankle' in keypoints_3d:
                # Calculate knee angle
                knee_angle = self.calculate_angle(
                    keypoints_3d['right_hip'],
                    keypoints_3d['right_knee'],
                    keypoints_3d['right_ankle']
                )
                feedback['right_knee'] = {
                    'value': knee_angle,
                    'ideal': key_angles['knee_angle']['ideal'],
                    'status': 'good' if abs(knee_angle - key_angles['knee_angle']['ideal']) <= key_angles['knee_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_knee']['status'] == 'good' else 0.5)
            
            if 'right_elbow' in keypoints_3d and 'right_shoulder' in keypoints_3d and 'right_hip' in keypoints_3d:
                # Calculate shoulder angle
                shoulder_angle = self.calculate_angle(
                    keypoints_3d['right_elbow'],
                    keypoints_3d['right_shoulder'],
                    keypoints_3d['right_hip']
                )
                feedback['right_shoulder'] = {
                    'value': shoulder_angle,
                    'ideal': key_angles['shoulder_angle']['ideal'],
                    'status': 'good' if abs(shoulder_angle - key_angles['shoulder_angle']['ideal']) <= key_angles['shoulder_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_shoulder']['status'] == 'good' else 0.5)
            
            if 'right_shoulder' in keypoints_3d and 'right_hip' in keypoints_3d and 'right_knee' in keypoints_3d:
                # Calculate hip angle
                hip_angle = self.calculate_angle(
                    keypoints_3d['right_shoulder'],
                    keypoints_3d['right_hip'],
                    keypoints_3d['right_knee']
                )
                feedback['right_hip'] = {
                    'value': hip_angle,
                    'ideal': key_angles['hip_angle']['ideal'],
                    'status': 'good' if abs(hip_angle - key_angles['hip_angle']['ideal']) <= key_angles['hip_angle']['tolerance'] else 'needs_improvement'
                }
                scores.append(1.0 if feedback['right_hip']['status'] == 'good' else 0.5)
            
            # Calculate overall score
            overall_score = np.mean(scores) if scores else 0.0
            
            return feedback, overall_score
            
        except Exception as e:
            print(f"Error analyzing pose: {e}")
            return {}, 0.0

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points.
        
        Args:
            a, b, c: Points in 3D space (lists or numpy arrays)
            
        Returns:
            angle: Angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)




