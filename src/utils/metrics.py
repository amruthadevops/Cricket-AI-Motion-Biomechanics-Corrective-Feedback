# src/utils/metrics.py
import numpy as np
from typing import Dict, List, Tuple, Optional

def calculate_angle(point1: Dict, point2: Dict) -> float:
    """
    Calculate the angle of the line between two points.
    
    Args:
        point1: Dictionary with 'x' and 'y' coordinates
        point2: Dictionary with 'x' and 'y' coordinates
        
    Returns:
        Angle in degrees (0-180)
    """
    try:
        dx = point2["x"] - point1["x"]
        dy = point2["y"] - point1["y"]
        
        # Calculate the angle in radians
        angle_rad = np.arctan2(dy, dx)
        
        # Convert to degrees and normalize to 0-180
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 180
            
        return angle_deg
        
    except Exception as e:
        print(f"Error calculating angle: {str(e)}")
        return 0.0

def calculate_distance(point1: Dict, point2: Dict) -> float:
    """
    Calculate the distance between two points.
    
    Args:
        point1: Dictionary with 'x' and 'y' coordinates
        point2: Dictionary with 'x' and 'y' coordinates
        
    Returns:
        Distance between the points
    """
    try:
        dx = point2["x"] - point1["x"]
        dy = point2["y"] - point1["y"]
        
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance
        
    except Exception as e:
        print(f"Error calculating distance: {str(e)}")
        return 0.0

def calculate_arm_angle(shoulder: Dict, elbow: Dict, wrist: Dict) -> float:
    """
    Calculate the angle of the arm.
    
    Args:
        shoulder: Dictionary with 'x' and 'y' coordinates
        elbow: Dictionary with 'x' and 'y' coordinates
        wrist: Dictionary with 'x' and 'y' coordinates
        
    Returns:
        Angle in degrees
    """
    try:
        # Calculate vectors
        shoulder_to_elbow = np.array([elbow["x"] - shoulder["x"], elbow["y"] - shoulder["y"]])
        elbow_to_wrist = np.array([wrist["x"] - elbow["x"], wrist["y"] - elbow["y"]])
        
        # Normalize vectors
        shoulder_to_elbow_norm = np.linalg.norm(shoulder_to_elbow)
        elbow_to_wrist_norm = np.linalg.norm(elbow_to_wrist)
        
        if shoulder_to_elbow_norm == 0 or elbow_to_wrist_norm == 0:
            return 0.0
            
        shoulder_to_elbow = shoulder_to_elbow / shoulder_to_elbow_norm
        elbow_to_wrist = elbow_to_wrist / elbow_to_wrist_norm
        
        # Calculate angle between vectors
        dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
        
    except Exception as e:
        print(f"Error calculating arm angle: {str(e)}")
        return 0.0

def calculate_torque(left_shoulder: Dict, right_shoulder: Dict, left_hip: Dict, right_hip: Dict) -> float:
    """
    Calculate the torque between shoulders and hips.
    
    Args:
        left_shoulder: Dictionary with 'x' and 'y' coordinates
        right_shoulder: Dictionary with 'x' and 'y' coordinates
        left_hip: Dictionary with 'x' and 'y' coordinates
        right_hip: Dictionary with 'x' and 'y' coordinates
        
    Returns:
        Torque value
    """
    try:
        # Calculate shoulder vector
        shoulder_vector = np.array([
            right_shoulder["x"] - left_shoulder["x"],
            right_shoulder["y"] - left_shoulder["y"]
        ])
        
        # Calculate hip vector
        hip_vector = np.array([
            right_hip["x"] - left_hip["x"],
            right_hip["y"] - left_hip["y"]
        ])
        
        # Normalize vectors
        shoulder_norm = np.linalg.norm(shoulder_vector)
        hip_norm = np.linalg.norm(hip_vector)
        
        if shoulder_norm == 0 or hip_norm == 0:
            return 0.0
            
        shoulder_vector = shoulder_vector / shoulder_norm
        hip_vector = hip_vector / hip_norm
        
        # Calculate angle between vectors
        dot_product = np.dot(shoulder_vector, hip_vector)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        # Calculate torque (simplified as the angle difference)
        torque = angle_deg
        
        return torque
        
    except Exception as e:
        print(f"Error calculating torque: {str(e)}")
        return 0.0

def estimate_shot_type(bat_angle: float, elbow_position: Dict, hip_shoulder_rotation: float, foot_placement: str) -> Tuple[str, float]:
    """
    Estimate the type of cricket shot based on various features.
    
    Args:
        bat_angle: Angle of the bat in degrees
        elbow_position: Dictionary with 'x' and 'y' coordinates of the elbow
        hip_shoulder_rotation: Rotation between hips and shoulders in degrees
        foot_placement: String describing the foot placement
        
    Returns:
        Tuple of (shot_type, confidence)
    """
    try:
        # Initialize scores for each shot type
        shot_scores = {
            "drive": 0.0,
            "cut": 0.0,
            "pull": 0.0,
            "sweep": 0.0
        }
        
        # Evaluate based on bat angle
        if 30 <= bat_angle <= 60:
            shot_scores["drive"] += 0.4
        elif 120 <= bat_angle <= 150:
            shot_scores["cut"] += 0.4
        elif 60 <= bat_angle <= 120:
            shot_scores["pull"] += 0.4
        elif 0 <= bat_angle <= 30:
            shot_scores["sweep"] += 0.4
        
        # Evaluate based on elbow position
        elbow_height = elbow_position["y"]
        if elbow_height < 0.5:  # Assuming normalized coordinates
            shot_scores["drive"] += 0.2
            shot_scores["cut"] += 0.2
        else:
            shot_scores["pull"] += 0.2
            shot_scores["sweep"] += 0.2
        
        # Evaluate based on hip-shoulder rotation
        if hip_shoulder_rotation < 30:
            shot_scores["drive"] += 0.2
        elif hip_shoulder_rotation > 60:
            shot_scores["cut"] += 0.2
            shot_scores["pull"] += 0.2
        else:
            shot_scores["sweep"] += 0.2
        
        # Evaluate based on foot placement
        if "frontfoot" in foot_placement:
            shot_scores["drive"] += 0.2
            shot_scores["sweep"] += 0.2
        elif "backfoot" in foot_placement:
            shot_scores["cut"] += 0.2
            shot_scores["pull"] += 0.2
        
        # Find the shot type with the highest score
        best_shot = max(shot_scores, key=shot_scores.get)
        confidence = shot_scores[best_shot]
        
        return best_shot, confidence
        
    except Exception as e:
        print(f"Error estimating shot type: {str(e)}")
        return "unknown", 0.0

def detect_follow_through_balance(frames_data: List[Dict]) -> Dict:
    """
    Detect balance during follow-through.
    
    Args:
        frames_data: List of dictionaries containing frame data with pose keypoints
        
    Returns:
        Dictionary with balance detection results
    """
    try:
        result = {
            "detected": False,
            "score": 0.0,
            "feedback": ""
        }
        
        if not frames_data:
            result["feedback"] = "No frame data provided"
            return result
        
        result["detected"] = True
        
        # Calculate center of gravity for each frame
        cog_positions = []
        for frame in frames_data:
            if not frame.get("pose_detected"):
                continue
                
            keypoints = frame.get("key_points", {})
            if not all(k in keypoints for k in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]):
                continue
            
            # Calculate center of gravity (simplified as average of key points)
            cog_x = (keypoints["left_hip"]["x"] + keypoints["right_hip"]["x"] + 
                    keypoints["left_shoulder"]["x"] + keypoints["right_shoulder"]["x"]) / 4
            cog_y = (keypoints["left_hip"]["y"] + keypoints["right_hip"]["y"] + 
                    keypoints["left_shoulder"]["y"] + keypoints["right_shoulder"]["y"]) / 4
            
            cog_positions.append({"x": cog_x, "y": cog_y})
        
        if len(cog_positions) < 2:
            result["feedback"] = "Insufficient data for balance analysis"
            return result
        
        # Calculate stability of center of gravity
        x_values = [pos["x"] for pos in cog_positions]
        y_values = [pos["y"] for pos in cog_positions]
        
        x_var = np.var(x_values)
        y_var = np.var(y_values)
        
        # Calculate stability (inverse of variance)
        x_stability = 1.0 / (1.0 + x_var)
        y_stability = 1.0 / (1.0 + y_var)
        

        # Overall stability is the average of x and y stability
        stability = (x_stability + y_stability) / 2
        
        result["score"] = stability
        
        # Generate feedback
        if stability > 0.8:
            result["feedback"] = "Excellent balance during follow-through"
        elif stability > 0.6:
            result["feedback"] = "Good balance during follow-through"
        elif stability > 0.4:
            result["feedback"] = "Average balance, could improve stability"
        else:
            result["feedback"] = "Poor balance, focus on stability during follow-through"
        
        return result
        
    except Exception as e:
        return {
            "detected": False,
            "score": 0.0,
            "feedback": f"Error detecting follow-through balance: {str(e)}"
        }

def calculate_center_of_gravity_shift(frames_data: List[Dict]) -> Dict:
    """
    Calculate the shift in center of gravity during a sequence of frames.
    
    Args:
        frames_data: List of dictionaries containing frame data with pose keypoints
        
    Returns:
        Dictionary with COG shift analysis results
    """
    try:
        result = {
            "detected": False,
            "shift": 0.0,
            "direction": "unknown",
            "score": 0.0,
            "feedback": ""
        }
        
        if not frames_data:
            result["feedback"] = "No frame data provided"
            return result
        
        result["detected"] = True
        
        # Calculate center of gravity for each frame
        cog_positions = []
        for frame in frames_data:
            if not frame.get("pose_detected"):
                continue
                
            keypoints = frame.get("key_points", {})
            if not all(k in keypoints for k in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]):
                continue
            
            # Calculate center of gravity (simplified as average of key points)
            cog_x = (keypoints["left_hip"]["x"] + keypoints["right_hip"]["x"] + 
                    keypoints["left_shoulder"]["x"] + keypoints["right_shoulder"]["x"]) / 4
            cog_y = (keypoints["left_hip"]["y"] + keypoints["right_hip"]["y"] + 
                    keypoints["left_shoulder"]["y"] + keypoints["right_shoulder"]["y"]) / 4
            
            cog_positions.append({"x": cog_x, "y": cog_y})
        
        if len(cog_positions) < 2:
            result["feedback"] = "Insufficient data for COG shift analysis"
            return result
        
        # Calculate COG shift
        start_cog = cog_positions[0]
        end_cog = cog_positions[-1]
        
        dx = end_cog["x"] - start_cog["x"]
        dy = end_cog["y"] - start_cog["y"]
        
        shift = np.sqrt(dx**2 + dy**2)
        result["shift"] = shift
        
        # Determine direction
        if abs(dx) > abs(dy):
            result["direction"] = "horizontal" if dx > 0 else "horizontal_left"
        else:
            result["direction"] = "vertical_down" if dy > 0 else "vertical_up"
        
        # Calculate score based on shift magnitude
        # For batting/bowling, a moderate shift is good, too much or too little is bad
        ideal_shift_range = (0.1, 0.3)  # Assuming normalized coordinates
        
        if ideal_shift_range[0] <= shift <= ideal_shift_range[1]:
            result["score"] = 1.0
            result["feedback"] = f"Good COG shift ({shift:.3f}) in {result['direction']} direction"
        elif shift < ideal_shift_range[0]:
            result["score"] = shift / ideal_shift_range[0]
            result["feedback"] = f"Insufficient COG shift ({shift:.3f}), need more weight transfer"
        else:
            result["score"] = max(0, 1.0 - (shift - ideal_shift_range[1]) / ideal_shift_range[1])
            result["feedback"] = f"Excessive COG shift ({shift:.3f}), focus on balance"
        
        return result
        
    except Exception as e:
        return {
            "detected": False,
            "shift": 0.0,
            "direction": "unknown",
            "score": 0.0,
            "feedback": f"Error calculating COG shift: {str(e)}"
        }

def analyze_wrist_motion_path(wrist_positions: List[Dict]) -> Dict:
    """
    Analyze the path of wrist motion during a sequence of frames.
    
    Args:
        wrist_positions: List of dictionaries with 'x' and 'y' coordinates of wrist positions
        
    Returns:
        Dictionary with wrist motion path analysis results
    """
    try:
        result = {
            "detected": False,
            "smoothness": 0.0,
            "extension": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        if not wrist_positions or len(wrist_positions) < 3:
            result["feedback"] = "Insufficient wrist position data"
            return result
        
        result["detected"] = True
        
        # Calculate smoothness of wrist motion
        velocities = []
        for i in range(1, len(wrist_positions)):
            dx = wrist_positions[i]["x"] - wrist_positions[i-1]["x"]
            dy = wrist_positions[i]["y"] - wrist_positions[i-1]["y"]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = abs(velocities[i] - velocities[i-1])
            accelerations.append(acceleration)
        
        # Calculate smoothness (inverse of acceleration variance)
        if accelerations:
            accel_variance = np.var(accelerations)
            result["smoothness"] = 1.0 / (1.0 + accel_variance)
        else:
            result["smoothness"] = 0.0
        
        # Calculate wrist extension
        total_distance = 0
        for i in range(1, len(wrist_positions)):
            dx = wrist_positions[i]["x"] - wrist_positions[i-1]["x"]
            dy = wrist_positions[i]["y"] - wrist_positions[i-1]["y"]
            distance = np.sqrt(dx**2 + dy**2)
            total_distance += distance
        
        # Calculate direct distance from start to end
        start_pos = wrist_positions[0]
        end_pos = wrist_positions[-1]
        direct_distance = np.sqrt(
            (end_pos["x"] - start_pos["x"])**2 + 
            (end_pos["y"] - start_pos["y"])**2
        )
        
        # Calculate extension ratio (total distance / direct distance)
        if direct_distance > 0:
            extension_ratio = total_distance / direct_distance
            # Normalize to 0-1 range
            result["extension"] = min(1.0, extension_ratio / 3.0)
        else:
            result["extension"] = 0.0
        
        # Calculate overall score
        result["score"] = (result["smoothness"] + result["extension"]) / 2
        
        # Generate feedback
        if result["score"] > 0.8:
            result["feedback"] = "Excellent wrist motion with good smoothness and extension"
        elif result["score"] > 0.6:
            result["feedback"] = "Good wrist motion, could improve smoothness or extension"
        elif result["score"] > 0.4:
            result["feedback"] = "Average wrist motion, needs improvement in both smoothness and extension"
        else:
            result["feedback"] = "Poor wrist motion, focus on technique"
        
        return result
        
    except Exception as e:
        return {
            "detected": False,
            "smoothness": 0.0,
            "extension": 0.0,
            "score": 0.0,
            "feedback": f"Error analyzing wrist motion path: {str(e)}"
        }

def estimate_shoulder_hip_torque_transfer(frames_data: List[Dict]) -> Dict:
    """
    Estimate the efficiency of shoulder-hip torque transfer during a sequence of frames.
    
    Args:
        frames_data: List of dictionaries containing frame data with pose keypoints
        
    Returns:
        Dictionary with torque transfer analysis results
    """
    try:
        result = {
            "detected": False,
            "efficiency": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        if not frames_data:
            result["feedback"] = "No frame data provided"
            return result
        
        result["detected"] = True
        
        # Calculate shoulder-hip torque for each frame
        torque_values = []
        for frame in frames_data:
            if not frame.get("pose_detected"):
                continue
                
            keypoints = frame.get("key_points", {})
            if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                continue
            
            # Calculate shoulder-hip torque
            torque = calculate_torque(
                keypoints["left_shoulder"], 
                keypoints["right_shoulder"],
                keypoints["left_hip"], 
                keypoints["right_hip"]
            )
            torque_values.append(torque)
        
        if len(torque_values) < 2:
            result["feedback"] = "Insufficient data for torque analysis"
            return result
        
        # Calculate the rate of change of torque
        torque_changes = []
        for i in range(1, len(torque_values)):
            change = abs(torque_values[i] - torque_values[i-1])
            torque_changes.append(change)
        
        # Calculate efficiency (inverse of variance in torque changes)
        if torque_changes:
            change_variance = np.var(torque_changes)
            result["efficiency"] = 1.0 / (1.0 + change_variance)
        else:
            result["efficiency"] = 0.0
        
        result["score"] = result["efficiency"]
        
        # Generate feedback
        if result["score"] > 0.8:
            result["feedback"] = "Excellent shoulder-hip torque transfer efficiency"
        elif result["score"] > 0.6:
            result["feedback"] = "Good torque transfer, could be more consistent"
        elif result["score"] > 0.4:
            result["feedback"] = "Average torque transfer, work on consistency"
        else:
            result["feedback"] = "Poor torque transfer, focus on hip-shoulder separation"
        
        return result
        
    except Exception as e:
        return {
            "detected": False,
            "efficiency": 0.0,
            "score": 0.0,
            "feedback": f"Error estimating shoulder-hip torque transfer: {str(e)}"
        }

def calculate_bat_speed(prev_wrist_positions: List[Dict], curr_wrist_positions: List[Dict]) -> float:
    """
    Calculate the speed of the bat based on wrist positions.
    
    Args:
        prev_wrist_positions: List of previous wrist positions
        curr_wrist_positions: List of current wrist positions
        
    Returns:
        Bat speed value
    """
    try:
        if not prev_wrist_positions or not curr_wrist_positions:
            return 0.0
        
        # Calculate previous bat center
        prev_left_wrist = prev_wrist_positions[0]
        prev_right_wrist = prev_wrist_positions[1]
        prev_bat_center = {
            "x": (prev_left_wrist["x"] + prev_right_wrist["x"]) / 2,
            "y": (prev_left_wrist["y"] + prev_right_wrist["y"]) / 2
        }
        
        # Calculate current bat center
        curr_left_wrist = curr_wrist_positions[0]
        curr_right_wrist = curr_wrist_positions[1]
        curr_bat_center = {
            "x": (curr_left_wrist["x"] + curr_right_wrist["x"]) / 2,
            "y": (curr_left_wrist["y"] + curr_right_wrist["y"]) / 2
        }
        
        # Calculate distance moved
        dx = curr_bat_center["x"] - prev_bat_center["x"]
        dy = curr_bat_center["y"] - prev_bat_center["y"]
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance
        
    except Exception as e:
        print(f"Error calculating bat speed: {str(e)}")
        return 0.0

    

