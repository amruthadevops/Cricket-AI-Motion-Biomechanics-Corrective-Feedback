# src/analyzers/batting_analyzer.py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class BattingAnalyzer:
    def __init__(self):
        # Initialize thresholds and parameters
        self.min_confidence = 0.5
        self.trigger_movement_threshold = 0.1  # Normalized movement threshold
        self.bat_angle_thresholds = {
            "drive": (30, 60),
            "cut": (120, 150),
            "pull": (60, 120),
            "sweep": (0, 30)
        }
        
    def analyze(self, frames_data: List[Dict]) -> Dict:
        """
        Enhanced batting analysis with trigger movement detection and shot classification.
        
        Args:
            frames_data: List of dictionaries containing frame data with pose keypoints
            
        Returns:
            Dictionary containing batting analysis results
        """
        results = {
            "trigger_movement": {"type": "unknown", "score": 0.0, "feedback": ""},
            "shot_type": {"type": "unknown", "confidence": 0.0, "feedback": ""},
            "stance": {"alignment": "unknown", "score": 0.0, "feedback": ""},
            "timing": {"head_stability": True, "score": 0.0, "feedback": ""},
            "bat_angle": {"angle": 0.0, "score": 0.0, "feedback": ""},
            "foot_placement": {"position": "unknown", "score": 0.0, "feedback": ""},
            "scores": {
                "stance_score": 0.0,
                "timing_score": 0.0,
                "bat_angle_score": 0.0,
                "foot_placement_score": 0.0,
                "overall_score": 0.0
            },
            "corrections": [],
            "trajectory": []
        }
        
        if not frames_data:
            results["error"] = "No frame data provided"
            return results
            
        try:
            # Analyze trigger movement
            trigger_results = self._analyze_trigger_movement(frames_data)
            results["trigger_movement"] = trigger_results
            
            # Analyze shot type
            shot_results = self._classify_shot_type(frames_data)
            results["shot_type"] = shot_results
            
            # Analyze stance
            stance_results = self._analyze_stance(frames_data)
            results["stance"] = stance_results
            
            # Analyze timing
            timing_results = self._analyze_timing(frames_data)
            results["timing"] = timing_results
            
            # Analyze bat angle
            bat_angle_results = self._analyze_bat_angle(frames_data)
            results["bat_angle"] = bat_angle_results
            
            # Analyze foot placement
            foot_placement_results = self._analyze_foot_placement(frames_data)
            results["foot_placement"] = foot_placement_results
            
            # Calculate overall scores
            self._calculate_overall_scores(results)
            
            # Generate corrections
            results["corrections"] = self._generate_corrections(results)
            
            # Calculate bat trajectory
            results["trajectory"] = self._calculate_bat_trajectory(frames_data)
            
        except Exception as e:
            results["error"] = str(e)
            
        return results
    
    def _analyze_trigger_movement(self, frames_data: List[Dict]) -> Dict:
        """Analyze the batter's trigger movement before the shot."""
        result = {"type": "unknown", "score": 0.0, "feedback": ""}
        
        try:
            # Get the first few frames to analyze initial movement
            initial_frames = frames_data[:10] if len(frames_data) >= 10 else frames_data
            
            if not initial_frames or not all(f.get("pose_detected") for f in initial_frames):
                result["feedback"] = "Insufficient pose data for trigger movement analysis"
                return result
                
            # Extract hip and ankle positions
            left_hip_positions = []
            right_hip_positions = []
            left_ankle_positions = []
            right_ankle_positions = []
            
            for frame in initial_frames:
                keypoints = frame.get("key_points", {})
                if "left_hip" in keypoints and "right_hip" in keypoints:
                    left_hip_positions.append(keypoints["left_hip"])
                    right_hip_positions.append(keypoints["right_hip"])
                if "left_ankle" in keypoints and "right_ankle" in keypoints:
                    left_ankle_positions.append(keypoints["left_ankle"])
                    right_ankle_positions.append(keypoints["right_ankle"])
            
            if not left_hip_positions or not right_hip_positions or not left_ankle_positions or not right_ankle_positions:
                result["feedback"] = "Missing keypoints for trigger movement analysis"
                return result
            
            # Calculate movement vectors
            left_hip_movement = self._calculate_movement_vector(left_hip_positions)
            right_hip_movement = self._calculate_movement_vector(right_hip_positions)
            left_ankle_movement = self._calculate_movement_vector(left_ankle_positions)
            right_ankle_movement = self._calculate_movement_vector(right_ankle_positions)
            
            # Determine trigger movement type
            backfoot_movement = abs(left_ankle_movement[0]) + abs(right_ankle_movement[0])
            frontfoot_movement = abs(left_ankle_movement[1]) + abs(right_ankle_movement[1])
            hip_shift = abs(left_hip_movement[0]) + abs(right_hip_movement[0])
            
            if backfoot_movement > frontfoot_movement and backfoot_movement > self.trigger_movement_threshold:
                result["type"] = "backfoot"
                result["score"] = min(backfoot_movement * 5, 1.0)  # Scale to 0-1
                result["feedback"] = "Backfoot trigger movement detected"
            elif frontfoot_movement > self.trigger_movement_threshold:
                result["type"] = "frontfoot"
                result["score"] = min(frontfoot_movement * 5, 1.0)  # Scale to 0-1
                result["feedback"] = "Frontfoot trigger movement detected"
            elif hip_shift > self.trigger_movement_threshold:
                result["type"] = "neutral"
                result["score"] = min(hip_shift * 5, 1.0)  # Scale to 0-1
                result["feedback"] = "Neutral trigger movement with hip shift detected"
            else:
                result["feedback"] = "Minimal trigger movement detected"
                
        except Exception as e:
            result["feedback"] = f"Error analyzing trigger movement: {str(e)}"
            
        return result
    
    def _classify_shot_type(self, frames_data: List[Dict]) -> Dict:
        """Classify the type of shot played based on bat angle, elbow position, and body rotation."""
        result = {"type": "unknown", "confidence": 0.0, "feedback": ""}
        
        try:
            # Find the frame with maximum bat movement (likely the impact frame)
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                result["feedback"] = "Could not determine shot impact frame"
                return result
                
            impact_frame = frames_data[impact_frame_idx]
            keypoints = impact_frame.get("key_points", {})
            
            # Extract necessary keypoints
            if not all(k in keypoints for k in ["left_wrist", "right_wrist", "left_elbow", "right_elbow", 
                                               "left_shoulder", "right_shoulder", "left_hip", "right_hip",
                                               "left_ankle", "right_ankle"]):
                result["feedback"] = "Missing keypoints for shot classification"
                return result
            
            # Calculate bat angle
            left_wrist = keypoints["left_wrist"]
            right_wrist = keypoints["right_wrist"]
            bat_angle = self._calculate_bat_angle(left_wrist, right_wrist)
            
            # Calculate elbow position
            left_elbow = keypoints["left_elbow"]
            right_elbow = keypoints["right_elbow"]
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            
            # Calculate hip-shoulder rotation
            left_hip = keypoints["left_hip"]
            right_hip = keypoints["right_hip"]
            shoulder_hip_rotation = self._calculate_shoulder_hip_rotation(
                left_shoulder, right_shoulder, left_hip, right_hip
            )
            
            # Determine foot placement
            left_ankle = keypoints["left_ankle"]
            right_ankle = keypoints["right_ankle"]
            foot_placement = self._determine_foot_placement(left_ankle, right_ankle, shoulder_hip_rotation)
            
            # Classify shot type based on features
            shot_type, confidence = self._determine_shot_type(
                bat_angle, left_elbow, right_elbow, shoulder_hip_rotation, foot_placement
            )
            
            result["type"] = shot_type
            result["confidence"] = confidence
            result["feedback"] = f"Shot classified as {shot_type} with {confidence:.2f} confidence"
            
        except Exception as e:
            result["feedback"] = f"Error classifying shot type: {str(e)}"
            
        return result
    
    def _analyze_stance(self, frames_data: List[Dict]) -> Dict:
        """Analyze the batter's stance alignment."""
        result = {"alignment": "unknown", "score": 0.0, "feedback": ""}
        
        try:
            # Use the first frame for stance analysis
            if not frames_data or not frames_data[0].get("pose_detected"):
                result["feedback"] = "No valid pose data for stance analysis"
                return result
                
            first_frame = frames_data[0]
            keypoints = first_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                result["feedback"] = "Missing keypoints for stance analysis"
                return result
            
            # Extract keypoints
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            left_hip = keypoints["left_hip"]
            right_hip = keypoints["right_hip"]
            
            # Calculate shoulder-hip alignment
            shoulder_vector = np.array([right_shoulder["x"] - left_shoulder["x"], 
                                       right_shoulder["y"] - left_shoulder["y"]])
            hip_vector = np.array([right_hip["x"] - left_hip["x"], 
                                  right_hip["y"] - left_hip["y"]])
            
            # Normalize vectors
            shoulder_norm = np.linalg.norm(shoulder_vector)
            hip_norm = np.linalg.norm(hip_vector)
            
            if shoulder_norm == 0 or hip_norm == 0:
                result["feedback"] = "Invalid keypoints for stance analysis"
                return result
                
            shoulder_vector = shoulder_vector / shoulder_norm
            hip_vector = hip_vector / hip_norm
            
            # Calculate alignment score (dot product)
            alignment_score = np.dot(shoulder_vector, hip_vector)
            
            # Determine alignment quality
            if alignment_score > 0.95:
                result["alignment"] = "excellent"
                result["score"] = 1.0
                result["feedback"] = "Excellent hip-shoulder alignment"
            elif alignment_score > 0.85:
                result["alignment"] = "good"
                result["score"] = 0.8
                result["feedback"] = "Good hip-shoulder alignment"
            elif alignment_score > 0.7:
                result["alignment"] = "average"
                result["score"] = 0.6
                result["feedback"] = "Average hip-shoulder alignment, could improve"
            else:
                result["alignment"] = "poor"
                result["score"] = alignment_score
                result["feedback"] = "Improve hip-shoulder alignment for better stance"
                
        except Exception as e:
            result["feedback"] = f"Error analyzing stance: {str(e)}"
            
        return result
    
    def _analyze_timing(self, frames_data: List[Dict]) -> Dict:
        """Analyze the batter's timing based on head stability."""
        result = {"head_stability": True, "score": 0.0, "feedback": ""}
        
        try:
            # Get frames around the impact
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                result["feedback"] = "Could not determine shot impact frame"
                return result
                
            # Get frames before and after impact
            start_idx = max(0, impact_frame_idx - 5)
            end_idx = min(len(frames_data) - 1, impact_frame_idx + 5)
            relevant_frames = frames_data[start_idx:end_idx + 1]
            
            # Extract head positions
            head_positions = []
            for frame in relevant_frames:
                if frame.get("pose_detected") and "nose" in frame.get("key_points", {}):
                    head_positions.append(frame["key_points"]["nose"])
            
            if len(head_positions) < 3:
                result["feedback"] = "Insufficient head position data for timing analysis"
                return result
            
            # Calculate head movement
            head_movement = self._calculate_head_movement(head_positions)
            
            # Determine timing quality
            if head_movement < 5:  # Threshold for minimal head movement
                result["head_stability"] = True
                result["score"] = 1.0
                result["feedback"] = "Good head stability during shot"
            else:
                result["head_stability"] = False
                result["score"] = max(0, 1.0 - (head_movement / 20))  # Scale score based on movement
                result["feedback"] = f"Head movement detected during shot ({head_movement:.1f}px), focus on keeping head still"
                
        except Exception as e:
            result["feedback"] = f"Error analyzing timing: {str(e)}"
            
        return result

    def _analyze_bat_angle(self, frames_data: List[Dict]) -> Dict:
        """Analyze the bat angle at the point of impact."""
        result = {"angle": 0.0, "score": 0.0, "feedback": ""}
        
        try:
            # Find the impact frame
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                result["feedback"] = "Could not determine shot impact frame"
                return result
                
            impact_frame = frames_data[impact_frame_idx]
            keypoints = impact_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in keypoints for k in ["left_wrist", "right_wrist"]):
                result["feedback"] = "Missing wrist keypoints for bat angle analysis"
                return result
            
            # Calculate bat angle
            left_wrist = keypoints["left_wrist"]
            right_wrist = keypoints["right_wrist"]
            bat_angle = self._calculate_bat_angle(left_wrist, right_wrist)
            result["angle"] = bat_angle
            
            # Get shot type for angle evaluation
            shot_type = self._classify_shot_type_at_impact(frames_data, impact_frame_idx)
            
            # Define ideal bat angle ranges for each shot type
            ideal_ranges = {
                "drive": (45, 60),
                "cut": (120, 150),
                "pull": (60, 120),
                "sweep": (0, 30),
                "defensive": (90, 120)
            }
            
            # Evaluate bat angle based on shot type
            if shot_type in ideal_ranges:
                min_angle, max_angle = ideal_ranges[shot_type]
                
                if min_angle <= bat_angle <= max_angle:
                    result["score"] = 1.0
                    result["feedback"] = f"✅ Optimal bat angle ({bat_angle:.1f}°) for {shot_type}"
                else:
                    # Calculate how far outside the ideal range
                    if bat_angle < min_angle:
                        deviation = min_angle - bat_angle
                        result["feedback"] = f"❌ Bat too flat ({bat_angle:.1f}°), ideal range {min_angle}°-{max_angle}°"
                    else:
                        deviation = bat_angle - max_angle
                        result["feedback"] = f"❌ Bat too steep ({bat_angle:.1f}°), ideal range {min_angle}°-{max_angle}°"
                    
                    # Score decreases with deviation
                    result["score"] = max(0, 1.0 - (deviation / 30))
            else:
                # Default evaluation for unknown shot types
                if 30 <= bat_angle <= 150:
                    result["score"] = 0.7
                    result["feedback"] = f"⚠ Acceptable bat angle ({bat_angle:.1f}°)"
                else:
                    result["score"] = 0.3
                    result["feedback"] = f"❌ Poor bat angle ({bat_angle:.1f}°) outside typical range"
                
        except Exception as e:
            result["feedback"] = f"Error analyzing bat angle: {str(e)}"
            
        return result

    
    def _analyze_foot_placement(self, frames_data: List[Dict]) -> Dict:
        """Analyze the batter's foot placement during the shot."""
        result = {"position": "unknown", "score": 0.0, "feedback": ""}
        
        try:
            # Find the impact frame
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                result["feedback"] = "Could not determine shot impact frame"
                return result
                
            impact_frame = frames_data[impact_frame_idx]
            keypoints = impact_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in keypoints for k in ["left_ankle", "right_ankle", "left_hip", "right_hip"]):
                result["feedback"] = "Missing keypoints for foot placement analysis"
                return result
            
            # Extract keypoints
            left_ankle = keypoints["left_ankle"]
            right_ankle = keypoints["right_ankle"]
            left_hip = keypoints["left_hip"]
            right_hip = keypoints["right_hip"]
            
            # Calculate body orientation
            hip_center = {"x": (left_hip["x"] + right_hip["x"]) / 2, "y": (left_hip["y"] + right_hip["y"]) / 2}
            
            # Determine foot placement relative to body
            left_foot_forward = left_ankle["y"] < hip_center["y"]
            right_foot_forward = right_ankle["y"] < hip_center["y"]
            
            # Calculate foot width (distance between ankles)
            foot_width = np.sqrt((right_ankle["x"] - left_ankle["x"])**2 + (right_ankle["y"] - left_ankle["y"])**2)
            
            # Determine foot placement type
            if left_foot_forward and not right_foot_forward:
                result["position"] = "left_foot_forward"
                result["score"] = 0.9
                result["feedback"] = "Left foot forward position"
            elif right_foot_forward and not left_foot_forward:
                result["position"] = "right_foot_forward"
                result["score"] = 0.9
                result["feedback"] = "Right foot forward position"
            elif left_foot_forward and right_foot_forward:
                result["position"] = "both_feet_forward"
                result["score"] = 0.7
                result["feedback"] = "Both feet forward position"
            else:
                result["position"] = "side_on"
                result["score"] = 0.8
                result["feedback"] = "Side-on position"
            
            # Adjust score based on foot width
            ideal_foot_width = 50  # Adjust based on your scale
            width_ratio = min(foot_width / ideal_foot_width, 2.0)
            
            if 0.8 <= width_ratio <= 1.2:
                # Good foot width
                result["score"] = min(result["score"] * 1.1, 1.0)
                result["feedback"] += " with good width"
            elif width_ratio < 0.8:
                # Feet too close
                result["score"] *= 0.8
                result["feedback"] += " but feet too close together"
            else:
                # Feet too wide
                result["score"] *= 0.8
                result["feedback"] += " but stance too wide"
                
        except Exception as e:
            result["feedback"] = f"Error analyzing foot placement: {str(e)}"
            
        return result
    
    def _calculate_overall_scores(self, results: Dict) -> None:
        """Calculate overall scores from individual analysis components."""
        try:
            stance_score = results["stance"]["score"]
            timing_score = results["timing"]["score"]
            bat_angle_score = results["bat_angle"]["score"]
            foot_placement_score = results["foot_placement"]["score"]
            
            # Calculate weighted average
            overall_score = (
                stance_score * 0.25 +
                timing_score * 0.25 +
                bat_angle_score * 0.3 +
                foot_placement_score * 0.2
            )
            
            # Update scores dictionary
            results["scores"]["stance_score"] = stance_score
            results["scores"]["timing_score"] = timing_score
            results["scores"]["bat_angle_score"] = bat_angle_score
            results["scores"]["foot_placement_score"] = foot_placement_score
            results["scores"]["overall_score"] = overall_score
            
        except Exception as e:
            print(f"Error calculating overall scores: {str(e)}")
    
    def _generate_corrections(self, results: Dict) -> List[str]:
        """Generate corrective feedback based on analysis results."""
        corrections = []
        
        try:
            # Trigger movement corrections
            if results["trigger_movement"]["type"] == "unknown":
                corrections.append("Work on developing a consistent trigger movement")
            
            # Stance corrections
            if results["stance"]["alignment"] == "poor":
                corrections.append("Improve hip-shoulder alignment in your stance")
            elif results["stance"]["alignment"] == "average":
                corrections.append("Refine your stance alignment for better stability")
            
            # Timing corrections
            if not results["timing"]["head_stability"]:
                corrections.append("Keep your head still during the shot for better timing")
            
            # Bat angle corrections
            if results["bat_angle"]["score"] < 0.7:
                corrections.append(f"Adjust bat angle: {results['bat_angle']['feedback']}")
            
            # Foot placement corrections
            if results["foot_placement"]["score"] < 0.7:
                corrections.append(f"Improve foot placement: {results['foot_placement']['feedback']}")
            
            # Shot type specific corrections
            shot_type = results["shot_type"]["type"]
            if shot_type != "unknown" and results["shot_type"]["confidence"] > 0.5:
                if shot_type == "drive" and results["bat_angle"]["score"] < 0.8:
                    corrections.append("For drives, ensure a more vertical bat angle")
                elif shot_type == "cut" and results["bat_angle"]["angle"] < 120:
                    corrections.append("For cut shots, ensure a more horizontal bat angle")
                elif shot_type == "pull" and results["foot_placement"]["position"] != "backfoot":
                    corrections.append("For pull shots, get onto the backfoot")
                elif shot_type == "sweep" and results["foot_placement"]["position"] != "frontfoot":
                    corrections.append("For sweep shots, get onto the frontfoot")
            
            # If no specific corrections, provide general encouragement
            if not corrections:
                corrections.append("Good technique overall - continue practicing!")
                
        except Exception as e:
            corrections.append(f"Error generating corrections: {str(e)}")
            
        return corrections
    
    def _calculate_bat_trajectory(self, frames_data: List[Dict]) -> List[Dict]:
        """Calculate the trajectory of the bat during the shot."""
        trajectory = []
        
        try:
            # Find the start and end of the shot
            start_idx = self._find_shot_start(frames_data)
            end_idx = self._find_shot_end(frames_data)
            
            if start_idx is None or end_idx is None:
                return trajectory
            
            # Extract bat positions for each frame in the shot
            for i in range(start_idx, end_idx + 1):
                frame = frames_data[i]
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if not all(k in keypoints for k in ["left_wrist", "right_wrist"]):
                    continue
                
                # Calculate bat center position
                left_wrist = keypoints["left_wrist"]
                right_wrist = keypoints["right_wrist"]
                bat_center = {
                    "x": (left_wrist["x"] + right_wrist["x"]) / 2,
                    "y": (left_wrist["y"] + right_wrist["y"]) / 2
                }
                
                # Calculate bat angle
                bat_angle = self._calculate_bat_angle(left_wrist, right_wrist)
                
                # Add to trajectory
                trajectory.append({
                    "frame": i,
                    "position": bat_center,
                    "angle": bat_angle
                })
                
        except Exception as e:
            print(f"Error calculating bat trajectory: {str(e)}")
            
        return trajectory
    
    def _calculate_movement_vector(self, positions: List[Dict]) -> Tuple[float, float]:
        """Calculate the movement vector between the first and last positions."""
        if len(positions) < 2:
            return (0.0, 0.0)
            
        first_pos = positions[0]
        last_pos = positions[-1]
        
        dx = last_pos["x"] - first_pos["x"]
        dy = last_pos["y"] - first_pos["y"]
        
        return (dx, dy)
    
    def _find_impact_frame(self, frames_data: List[Dict]) -> Optional[int]:
        """Find the frame where the bat impacts the ball."""
        try:
            # Look for the frame with maximum bat speed
            max_speed = 0
            impact_frame_idx = None
            
            for i in range(1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_wrist", "right_wrist"]) or \
                   not all(k in curr_keypoints for k in ["left_wrist", "right_wrist"]):
                    continue
                
                # Calculate bat speed
                prev_left_wrist = prev_keypoints["left_wrist"]
                prev_right_wrist = prev_keypoints["right_wrist"]
                curr_left_wrist = curr_keypoints["left_wrist"]
                curr_right_wrist = curr_keypoints["right_wrist"]
                
                prev_bat_center = {
                    "x": (prev_left_wrist["x"] + prev_right_wrist["x"]) / 2,
                    "y": (prev_left_wrist["y"] + prev_right_wrist["y"]) / 2
                }
                
                curr_bat_center = {
                    "x": (curr_left_wrist["x"] + curr_right_wrist["x"]) / 2,
                    "y": (curr_left_wrist["y"] + curr_right_wrist["y"]) / 2
                }
                
                # Calculate distance moved
                distance = np.sqrt(
                    (curr_bat_center["x"] - prev_bat_center["x"])**2 + 
                    (curr_bat_center["y"] - prev_bat_center["y"])**2
                )
                
                if distance > max_speed:
                    max_speed = distance
                    impact_frame_idx = i
            
            return impact_frame_idx
            
        except Exception as e:
            print(f"Error finding impact frame: {str(e)}")
            return None
    
    def _find_shot_start(self, frames_data: List[Dict]) -> Optional[int]:
        """Find the frame where the shot starts."""
        try:
            # Look for the frame where the bat starts moving significantly
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                return None
                
            # Look backwards from the impact frame
            for i in range(impact_frame_idx - 1, 0, -1):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_wrist", "right_wrist"]) or \
                   not all(k in curr_keypoints for k in ["left_wrist", "right_wrist"]):
                    continue
                
                # Calculate bat speed
                prev_left_wrist = prev_keypoints["left_wrist"]
                prev_right_wrist = prev_keypoints["right_wrist"]
                curr_left_wrist = curr_keypoints["left_wrist"]
                curr_right_wrist = curr_keypoints["right_wrist"]
                
                prev_bat_center = {
                    "x": (prev_left_wrist["x"] + prev_right_wrist["x"]) / 2,
                    "y": (prev_left_wrist["y"] + prev_right_wrist["y"]) / 2
                }
                
                curr_bat_center = {
                    "x": (curr_left_wrist["x"] + curr_right_wrist["x"]) / 2,
                    "y": (curr_left_wrist["y"] + curr_right_wrist["y"]) / 2
                }
                
                # Calculate distance moved
                distance = np.sqrt(
                    (curr_bat_center["x"] - prev_bat_center["x"])**2 + 
                    (curr_bat_center["y"] - prev_bat_center["y"])**2
                )
                
                # If the bat speed drops below a threshold, we've found the start
                if distance < 2.0:  # Threshold for minimal movement
                    return i
            
            # If we didn't find a clear start, return the first frame
            return 0
            
        except Exception as e:
            print(f"Error finding shot start: {str(e)}")
            return None
    
    def _find_shot_end(self, frames_data: List[Dict]) -> Optional[int]:
        """Find the frame where the shot ends."""
        try:
            # Look for the frame where the bat stops moving significantly
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                return None
                
            # Look forwards from the impact frame
            for i in range(impact_frame_idx + 1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_wrist", "right_wrist"]) or \
                   not all(k in curr_keypoints for k in ["left_wrist", "right_wrist"]):
                    continue
                
                # Calculate bat speed
                prev_left_wrist = prev_keypoints["left_wrist"]
                prev_right_wrist = prev_keypoints["right_wrist"]
                curr_left_wrist = curr_keypoints["left_wrist"]
                curr_right_wrist = curr_keypoints["right_wrist"]
                
                prev_bat_center = {
                    "x": (prev_left_wrist["x"] + prev_right_wrist["x"]) / 2,
                    "y": (prev_left_wrist["y"] + prev_right_wrist["y"]) / 2
                }
                
                curr_bat_center = {
                    "x": (curr_left_wrist["x"] + curr_right_wrist["x"]) / 2,
                    "y": (curr_left_wrist["y"] + curr_right_wrist["y"]) / 2
                }
                
                # Calculate distance moved
                distance = np.sqrt(
                    (curr_bat_center["x"] - prev_bat_center["x"])**2 + 
                    (curr_bat_center["y"] - prev_bat_center["y"])**2
                )
                
                # If the bat speed drops below a threshold, we've found the end
                if distance < 2.0:  # Threshold for minimal movement
                    return i
            
            # If we didn't find a clear end, return the last frame
            return len(frames_data) - 1
            
        except Exception as e:
            print(f"Error finding shot end: {str(e)}")
            return None
    
    def _calculate_bat_angle(self, left_wrist: Dict, right_wrist: Dict) -> float:
        """Calculate the angle of the bat in degrees."""
        try:
            # Calculate the vector from left wrist to right wrist
            dx = right_wrist["x"] - left_wrist["x"]
            dy = right_wrist["y"] - left_wrist["y"]
            
            # Calculate the angle in radians
            angle_rad = np.arctan2(dy, dx)
            
            # Convert to degrees and normalize to 0-180
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 180
                
            return angle_deg
            
        except Exception as e:
            print(f"Error calculating bat angle: {str(e)}")
            return 0.0
    
    def _calculate_shoulder_hip_rotation(self, left_shoulder: Dict, right_shoulder: Dict, 
                                       left_hip: Dict, right_hip: Dict) -> float:
        """Calculate the rotation between shoulders and hips."""
        try:
            # Calculate shoulder vector
            shoulder_vector = np.array([right_shoulder["x"] - left_shoulder["x"], 
                                       right_shoulder["y"] - left_shoulder["y"]])
            
            # Calculate hip vector
            hip_vector = np.array([right_hip["x"] - left_hip["x"], 
                                  right_hip["y"] - left_hip["y"]])
            
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
            
            return angle_deg
            
        except Exception as e:
            print(f"Error calculating shoulder-hip rotation: {str(e)}")
            return 0.0
    
    def _determine_foot_placement(self, left_ankle: Dict, right_ankle: Dict, 
                                shoulder_hip_rotation: float) -> str:
        """Determine the foot placement based on ankle positions and body rotation."""
        try:
            # Calculate the center point between ankles
            ankle_center_x = (left_ankle["x"] + right_ankle["x"]) / 2
            
            # Determine which foot is forward based on y-coordinates
            if left_ankle["y"] < right_ankle["y"]:
                forward_foot = "left"
            else:
                forward_foot = "right"
            
            # Determine if it's leg-side or off-side based on rotation
            if shoulder_hip_rotation < 45:
                side = "off"
            else:
                side = "leg"
            
            # Combine to get foot placement
            if forward_foot == "left":
                return f"left_foot_{side}_side"
            else:
                return f"right_foot_{side}_side"
                
        except Exception as e:
            print(f"Error determining foot placement: {str(e)}")
            return "unknown"
    
    def _determine_shot_type(self, bat_angle: float, left_elbow: Dict, right_elbow: Dict,
                           shoulder_hip_rotation: float, foot_placement: str) -> Tuple[str, float]:
        """Determine the shot type based on various features."""
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
            elbow_height = (left_elbow["y"] + right_elbow["y"]) / 2
            if elbow_height < 0.5:  # Assuming normalized coordinates
                shot_scores["drive"] += 0.2
                shot_scores["cut"] += 0.2
            else:
                shot_scores["pull"] += 0.2
                shot_scores["sweep"] += 0.2
            
            # Evaluate based on shoulder-hip rotation
            if shoulder_hip_rotation < 30:
                shot_scores["drive"]
                shot_scores["drive"] += 0.2
            elif shoulder_hip_rotation > 60:
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
            print(f"Error determining shot type: {str(e)}")
            return "unknown", 0.0
    
    def _calculate_head_movement(self, head_positions: List[Dict]) -> float:
        """Calculate the total movement of the head."""
        try:
            if len(head_positions) < 2:
                return 0.0
            
            # Calculate the total distance traveled by the head
            total_distance = 0
            for i in range(1, len(head_positions)):
                dx = head_positions[i]["x"] - head_positions[i-1]["x"]
                dy = head_positions[i]["y"] - head_positions[i-1]["y"]
                distance = np.sqrt(dx**2 + dy**2)
                total_distance += distance
            
            return total_distance
            
        except Exception as e:
            print(f"Error calculating head movement: {str(e)}")
            return 0.0

    def _analyze_trigger_movement_relative_to_ball(self, frames_data: List[Dict]) -> Dict:
        """Analyze trigger movement relative to ball release."""
        result = {
            "type": "unknown",
            "timing": "unknown",
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find the ball release frame (approximated as impact frame)
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                result["feedback"] = "Could not determine ball release frame"
                return result
            
            # Analyze back foot movement before impact
            back_foot_positions = []
            for i in range(max(0, impact_frame_idx - 15), impact_frame_idx):
                if i >= len(frames_data):
                    continue
                    
                frame = frames_data[i]
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if "right_ankle" in keypoints:
                    back_foot_positions.append(keypoints["right_ankle"])
            
            if len(back_foot_positions) < 2:
                result["feedback"] = "Insufficient back foot position data"
                return result
            
            # Calculate back foot movement
            movement_vector = self._calculate_movement_vector(back_foot_positions)
            movement_magnitude = np.sqrt(movement_vector[0]**2 + movement_vector[1]**2)
            
            # Determine trigger timing based on when movement started
            trigger_frame = impact_frame_idx - len(back_foot_positions)
            frames_before_impact = impact_frame_idx - trigger_frame
            
            # Classify trigger timing
            if frames_before_impact > 10:
                result["timing"] = "early"
                result["score"] = 0.3
                result["feedback"] = "Early trigger movement - too much time before ball release"
            elif frames_before_impact > 5:
                result["timing"] = "ideal"
                result["score"] = 0.9
                result["feedback"] = "Ideal trigger timing - well synchronized with ball release"
            else:
                result["timing"] = "late"
                result["score"] = 0.5
                result["feedback"] = "Late trigger movement - rushed response to ball"
            
            # Classify trigger type based on movement direction
            if abs(movement_vector[0]) > abs(movement_vector[1]):
                result["type"] = "horizontal"
            else:
                result["type"] = "vertical"
            
            # Adjust score based on movement magnitude
            if 0.05 <= movement_magnitude <= 0.15:
                result["score"] = min(result["score"] * 1.2, 1.0)
            else:
                result["score"] = max(result["score"] * 0.8, 0.1)
            
        except Exception as e:
            result["feedback"] = f"Error analyzing trigger movement: {str(e)}"
            
        return result
    


    def _classify_shot_type_at_impact(self, frames_data: List[Dict], impact_frame_idx: int) -> str:
        """Classify shot type based on motion direction at impact."""
        try:
            if impact_frame_idx is None or impact_frame_idx >= len(frames_data):
                return "unknown"
            
            # Get frames before and after impact for motion analysis
            start_idx = max(0, impact_frame_idx - 5)
            end_idx = min(len(frames_data) - 1, impact_frame_idx + 5)
            
            # Analyze bat direction
            bat_directions = []
            for i in range(start_idx, end_idx):
                if i >= len(frames_data) - 1:
                    continue
                    
                frame = frames_data[i]
                next_frame = frames_data[i + 1]
                
                if not frame.get("pose_detected") or not next_frame.get("pose_detected"):
                    continue
                
                keypoints = frame.get("key_points", {})
                next_keypoints = next_frame.get("key_points", {})
                
                if not all(k in keypoints for k in ["left_wrist", "right_wrist"]) or \
                   not all(k in next_keypoints for k in ["left_wrist", "right_wrist"]):
                    continue
                
                # Calculate bat movement
                curr_bat_center = {
                    "x": (keypoints["left_wrist"]["x"] + keypoints["right_wrist"]["x"]) / 2,
                    "y": (keypoints["left_wrist"]["y"] + keypoints["right_wrist"]["y"]) / 2
                }
                
                next_bat_center = {
                    "x": (next_keypoints["left_wrist"]["x"] + next_keypoints["right_wrist"]["x"]) / 2,
                    "y": (next_keypoints["left_wrist"]["y"] + next_keypoints["right_wrist"]["y"]) / 2
                }
                
                dx = next_bat_center["x"] - curr_bat_center["x"]
                dy = next_bat_center["y"] - curr_bat_center["y"]
                
                # Calculate movement angle
                movement_angle = np.degrees(np.arctan2(dy, dx))
                if movement_angle < 0:
                    movement_angle += 180
                
                bat_directions.append(movement_angle)
            
            if not bat_directions:
                return "unknown"
            
            # Calculate average movement direction
            avg_direction = np.mean(bat_directions)
            
            # Classify shot type based on direction
            if 0 <= avg_direction < 45:
                return "drive"
            elif 45 <= avg_direction < 90:
                return "pull"
            elif 90 <= avg_direction < 135:
                return "cut"
            elif 135 <= avg_direction < 180:
                return "sweep"
            else:
                return "defensive"
                
        except Exception as e:
            print(f"Error classifying shot type: {str(e)}")
            return "unknown"

    def _analyze_batting_follow_through(self, frames_data: List[Dict]) -> Dict:
        """Analyze the batting follow-through to identify abrupt or incomplete follow-throughs."""
        result = {
            "quality": "unknown",
            "completeness": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find the impact frame
            impact_frame_idx = self._find_impact_frame(frames_data)
            if impact_frame_idx is None:
                result["feedback"] = "Could not determine shot impact frame"
                return result
            
            # Find the end of the shot
            end_frame_idx = self._find_shot_end(frames_data)
            if end_frame_idx is None:
                result["feedback"] = "Could not determine shot end frame"
                return result
            
            # Analyze follow-through frames
            follow_through_frames = frames_data[impact_frame_idx+1:end_frame_idx+1]
            
            if len(follow_through_frames) < 5:
                result["quality"] = "incomplete"
                result["completeness"] = len(follow_through_frames) / 10.0  # Normalize to 0-1
                result["score"] = result["completeness"]
                result["feedback"] = "❌ Incomplete follow-through - too short"
                return result
            
            # Calculate wrist movement smoothness during follow-through
            wrist_positions = []
            for frame in follow_through_frames:
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if "right_wrist" in keypoints:
                    wrist_positions.append(keypoints["right_wrist"])
            
            if len(wrist_positions) < 2:
                result["feedback"] = "Insufficient wrist position data for follow-through analysis"
                return result
            
            # Calculate wrist velocities
            velocities = []
            for i in range(1, len(wrist_positions)):
                dx = wrist_positions[i]["x"] - wrist_positions[i-1]["x"]
                dy = wrist_positions[i]["y"] - wrist_positions[i-1]["y"]
                velocity = np.sqrt(dx**2 + dy**2)
                velocities.append(velocity)
            
            # Calculate velocity changes (deceleration pattern)
            velocity_changes = []
            for i in range(1, len(velocities)):
                change = velocities[i] - velocities[i-1]
                velocity_changes.append(change)
            
            # Analyze follow-through quality based on velocity pattern
            if len(velocity_changes) > 0:
                # A good follow-through should have gradual deceleration
                avg_deceleration = np.mean([c for c in velocity_changes if c < 0])
                max_deceleration = min(velocity_changes)
                
                if abs(max_deceleration) > 0.1:  # Abrupt stop
                    result["quality"] = "abrupt"
                    result["score"] = 0.3
                    result["feedback"] = "❌ Abrupt follow-through - sudden stop"
                elif avg_deceleration < -0.02:  # Too rapid deceleration
                    result["quality"] = "poor"
                    result["score"] = 0.5
                    result["feedback"] = "❌ Poor follow-through - too rapid deceleration"
                else:
                    result["quality"] = "good"
                    result["score"] = 0.9
                    result["feedback"] = "✅ Good follow-through - smooth deceleration"
            else:
                result["quality"] = "unknown"
                result["score"] = 0.5
                result["feedback"] = "⚠ Could not determine follow-through quality"
            
            # Calculate completeness based on follow-through length
            ideal_length = 15  # frames
            actual_length = len(follow_through_frames)
            result["completeness"] = min(actual_length / ideal_length, 1.0)
            
            # Adjust score based on completeness
            if result["quality"] == "good":
                result["score"] = result["score"] * result["completeness"]
            
        except Exception as e:
            result["feedback"] = f"Error analyzing batting follow-through: {str(e)}"
            
        return result


