# src/analyzers/fielding_analyzer.py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class FieldingAnalyzer:
    def __init__(self):
        # Initialize thresholds and parameters
        self.min_confidence = 0.5
        self.dive_threshold = 0.2  # Threshold for detecting dive
        self.throw_angle_thresholds = {
            "good": (30, 60),
            "acceptable": (20, 70)
        }
        self.shoulder_rotation_threshold = 45  # Degrees
        self.wrist_extension_threshold = 0.15  # Normalized extension
        
    def analyze(self, frames_data: List[Dict]) -> Dict:
        """
        Enhanced fielding analysis with dive orientation and throw mechanics.
        
        Args:
            frames_data: List of dictionaries containing frame data with pose keypoints
            
        Returns:
            Dictionary containing fielding analysis results
        """
        results = {
            "dive_analysis": {
                "detected": False,
                "orientation": "unknown",
                "quality": "unknown",
                "start_frame": -1,
                "end_frame": -1,
                "score": 0.0,
                "feedback": ""
            },
            "throw_mechanics": {
                "detected": False,
                "shoulder_rotation": 0.0,
                "wrist_extension": 0.0,
                "arm_lag": False,
                "follow_through": False,
                "score": 0.0,
                "feedback": ""
            },
            "anticipation_reaction": {
                "detected": False,
                "reaction_time": 0.0,
                "score": 0.0,
                "feedback": ""
            },
            "scores": {
                "dive_score": 0.0,
                "throw_score": 0.0,
                "anticipation_score": 0.0,
                "overall_score": 0.0
            },
            "corrections": []
        }
        
        if not frames_data:
            results["error"] = "No frame data provided"
            return results
            
        try:
            # Analyze dive
            dive_results = self._analyze_dive(frames_data)
            results["dive_analysis"] = dive_results
            
            # Analyze throw mechanics
            throw_results = self._analyze_throw_mechanics(frames_data)
            results["throw_mechanics"] = throw_results
            
            # Analyze anticipation and reaction
            anticipation_results = self._analyze_anticipation_reaction(frames_data)
            results["anticipation_reaction"] = anticipation_results
            
            # Calculate overall scores
            self._calculate_overall_scores(results)
            
            # Generate corrections
            results["corrections"] = self._generate_corrections(results)
            
        except Exception as e:
            results["error"] = str(e)
            
        return results
    
    def _analyze_dive(self, frames_data: List[Dict]) -> Dict:
        """Analyze the dive orientation and quality."""
        dive_result = {
            "detected": False,
            "orientation": "unknown",
            "quality": "unknown",
            "start_frame": -1,
            "end_frame": -1,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find dive frames
            dive_start, dive_end = self._find_dive_frames(frames_data)
            
            if dive_start is None or dive_end is None:
                dive_result["feedback"] = "No dive detected"
                return dive_result
            
            dive_result["detected"] = True
            dive_result["start_frame"] = dive_start
            dive_result["end_frame"] = dive_end

            # Get dive frames for analysis
            dive_frames = frames_data[dive_start:dive_end+1]
            
            # Analyze dive orientation
            orientation_result = self._analyze_dive_orientation(dive_frames)
            dive_result["orientation"] = orientation_result["orientation"]
            dive_result["score"] = orientation_result["score"]
            
            # Analyze dive quality
            quality_result = self._analyze_dive_quality(dive_frames)
            dive_result["quality"] = quality_result["quality"]
            
            # Combine scores
            dive_result["score"] = (orientation_result["score"] + quality_result["score"]) / 2
            
            # Generate feedback
            if dive_result["score"] > 0.8:
                dive_result["feedback"] = f"Excellent {orientation_result['orientation']} dive with good form"
            elif dive_result["score"] > 0.6:
                dive_result["feedback"] = f"Good {orientation_result['orientation']} dive, {quality_result['feedback'].lower()}"
            else:
                dive_result["feedback"] = f"{orientation_result['feedback']} and {quality_result['feedback'].lower()}"
                
        except Exception as e:
            dive_result["feedback"] = f"Error analyzing dive: {str(e)}"
            
        return dive_result
    
    def _analyze_throw_mechanics(self, frames_data: List[Dict]) -> Dict:
        """Analyze the throw mechanics including shoulder rotation and wrist extension."""
        throw_result = {
            "detected": False,
            "shoulder_rotation": 0.0,
            "wrist_extension": 0.0,
            "arm_lag": False,
            "follow_through": False,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find throw frames
            throw_start, throw_end = self._find_throw_frames(frames_data)
            
            if throw_start is None or throw_end is None:
                throw_result["feedback"] = "No throw detected"
                return throw_result
            
            throw_result["detected"] = True
            
            # Get throw frames for analysis
            throw_frames = frames_data[throw_start:throw_end+1]
            
            # Analyze shoulder rotation
            shoulder_rotation = self._analyze_shoulder_rotation(throw_frames)
            throw_result["shoulder_rotation"] = shoulder_rotation
            
            # Analyze wrist extension
            wrist_extension = self._analyze_wrist_extension(throw_frames)
            throw_result["wrist_extension"] = wrist_extension
            
            # Detect arm lag
            arm_lag = self._detect_arm_lag(throw_frames)
            throw_result["arm_lag"] = arm_lag
            
            # Detect follow-through
            follow_through = self._detect_follow_through(throw_frames)
            throw_result["follow_through"] = follow_through
            
            # Calculate overall score
            shoulder_score = min(shoulder_rotation / self.shoulder_rotation_threshold, 1.0)
            wrist_score = min(wrist_extension / self.wrist_extension_threshold, 1.0)
            arm_lag_score = 0.0 if arm_lag else 1.0
            follow_through_score = 1.0 if follow_through else 0.0
            
            throw_result["score"] = (
                shoulder_score * 0.4 +
                wrist_score * 0.3 +
                arm_lag_score * 0.15 +
                follow_through_score * 0.15
            )
            
            # Generate feedback
            feedback_parts = []
            if shoulder_score < 0.7:
                feedback_parts.append("Insufficient shoulder rotation")
            if wrist_score < 0.7:
                feedback_parts.append("Weak wrist extension")
            if arm_lag:
                feedback_parts.append("Arm lag detected")
            if not follow_through:
                feedback_parts.append("Incomplete follow-through")
                
            if feedback_parts:
                throw_result["feedback"] = ", ".join(feedback_parts)
            else:
                throw_result["feedback"] = "Good throwing mechanics"
                
        except Exception as e:
            throw_result["feedback"] = f"Error analyzing throw mechanics: {str(e)}"
            
        return throw_result
    
    def _analyze_anticipation_reaction(self, frames_data: List[Dict]) -> Dict:
        """Analyze anticipation and reaction time."""
        anticipation_result = {
            "detected": False,
            "reaction_time": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find the frame where the ball becomes visible (simulated)
            ball_visible_frame = self._find_ball_visible_frame(frames_data)
            
            if ball_visible_frame is None:
                anticipation_result["feedback"] = "Could not determine ball visibility"
                return anticipation_result
            
            anticipation_result["detected"] = True
            
            # Find the frame where the fielder starts moving
            movement_start_frame = self._find_movement_start_frame(frames_data, ball_visible_frame)
            
            if movement_start_frame is None:
                anticipation_result["feedback"] = "No clear movement detected"
                return anticipation_result
            
            # Calculate reaction time (in frames)
            reaction_time = movement_start_frame - ball_visible_frame
            anticipation_result["reaction_time"] = reaction_time
            
            # Calculate score (lower reaction time is better)
            if reaction_time <= 3:
                anticipation_result["score"] = 1.0
                anticipation_result["feedback"] = "Excellent anticipation and quick reaction"
            elif reaction_time <= 6:
                anticipation_result["score"] = 0.7
                anticipation_result["feedback"] = "Good anticipation and reaction time"
            elif reaction_time <= 10:
                anticipation_result["score"] = 0.4
                anticipation_result["feedback"] = "Average reaction time, could improve anticipation"
            else:
                anticipation_result["score"] = 0.2
                anticipation_result["feedback"] = "Slow reaction time, work on anticipation"
                
        except Exception as e:
            anticipation_result["feedback"] = f"Error analyzing anticipation and reaction: {str(e)}"
            
        return anticipation_result
    
    def _calculate_overall_scores(self, results: Dict) -> None:
        """Calculate overall scores from individual analysis components."""
        try:
            # Get individual scores
            dive_score = results["dive_analysis"]["score"] if results["dive_analysis"]["detected"] else 0.5
            throw_score = results["throw_mechanics"]["score"] if results["throw_mechanics"]["detected"] else 0.5
            anticipation_score = results["anticipation_reaction"]["score"] if results["anticipation_reaction"]["detected"] else 0.5
            
            # Calculate weighted average
            overall_score = (
                dive_score * 0.4 +
                throw_score * 0.4 +
                anticipation_score * 0.2
            )
            
            # Update scores dictionary
            results["scores"]["dive_score"] = dive_score
            results["scores"]["throw_score"] = throw_score
            results["scores"]["anticipation_score"] = anticipation_score
            results["scores"]["overall_score"] = overall_score
            
        except Exception as e:
            print(f"Error calculating overall scores: {str(e)}")
    
    def _generate_corrections(self, results: Dict) -> List[str]:
        """Generate corrective feedback based on analysis results."""
        corrections = []
        
        try:
            # Dive corrections
            if results["dive_analysis"]["detected"]:
                if results["dive_analysis"]["score"] < 0.7:
                    if results["dive_analysis"]["orientation"] == "poor":
                        corrections.append("Improve dive orientation - extend arms and keep body parallel to ground")
                    if results["dive_analysis"]["quality"] == "poor":
                        corrections.append("Work on dive technique - absorb impact with arms and chest")
            else:
                corrections.append("Practice diving technique for fielding improvement")
            
            # Throw corrections
            if results["throw_mechanics"]["detected"]:
                if results["throw_mechanics"]["score"] < 0.7:
                    if results["throw_mechanics"]["shoulder_rotation"] < self.shoulder_rotation_threshold:
                        corrections.append("Increase shoulder rotation for more powerful throws")
                    if results["throw_mechanics"]["wrist_extension"] < self.wrist_extension_threshold:
                        corrections.append("Improve wrist extension for better ball control")
                    if results["throw_mechanics"]["arm_lag"]:
                        corrections.append("Eliminate arm lag for quicker release")
                    if not results["throw_mechanics"]["follow_through"]:
                        corrections.append("Complete follow-through for accurate throws")
            else:
                corrections.append("Practice throwing mechanics for better fielding")
            
            # Anticipation corrections
            if results["anticipation_reaction"]["detected"]:
                if results["anticipation_reaction"]["score"] < 0.7:
                    corrections.append("Work on anticipation and reaction time")
            else:
                corrections.append("Improve anticipation skills for better fielding")
            
            # If no specific corrections, provide general encouragement
            if not corrections:
                corrections.append("Good fielding technique overall - continue practicing!")
                
        except Exception as e:
            corrections.append(f"Error generating corrections: {str(e)}")
            
        return corrections
    
    def _find_dive_frames(self, frames_data: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
        """Find the start and end frames of a dive."""
        try:
            dive_start = None
            dive_end = None
            
            # Look for significant horizontal movement
            for i in range(1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_hip", "right_hip"]) or \
                   not all(k in curr_keypoints for k in ["left_hip", "right_hip"]):
                    continue
                
                # Calculate hip movement
                prev_hip_center = {
                    "x": (prev_keypoints["left_hip"]["x"] + prev_keypoints["right_hip"]["x"]) / 2,
                    "y": (prev_keypoints["left_hip"]["y"] + prev_keypoints["right_hip"]["y"]) / 2
                }
                
                curr_hip_center = {
                    "x": (curr_keypoints["left_hip"]["x"] + curr_keypoints["right_hip"]["x"]) / 2,
                    "y": (curr_keypoints["left_hip"]["y"] + curr_keypoints["right_hip"]["y"]) / 2
                }
                
                # Calculate horizontal movement
                dx = curr_hip_center["x"] - prev_hip_center["x"]
                dy = curr_hip_center["y"] - prev_hip_center["y"]
                
                # If horizontal movement is significant and greater than vertical movement, it might be a dive
                if abs(dx) > self.dive_threshold and abs(dx) > abs(dy):
                    if dive_start is None:
                        dive_start = i - 1
                    dive_end = i
            
            return dive_start, dive_end
            
        except Exception as e:
            print(f"Error finding dive frames: {str(e)}")
            return None, None
    
    def _analyze_dive_orientation(self, dive_frames: List[Dict]) -> Dict:
        """Analyze the orientation of the dive."""
        result = {"orientation": "unknown", "score": 0.0, "feedback": ""}
        
        try:
            if not dive_frames:
                result["feedback"] = "No dive frames provided"
                return result
            
            # Get the middle frame of the dive
            mid_frame = dive_frames[len(dive_frames) // 2]
            keypoints = mid_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                result["feedback"] = "Missing keypoints for dive orientation analysis"
                return result
            
            # Extract keypoints
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            left_hip = keypoints["left_hip"]
            right_hip = keypoints["right_hip"]
            
            # Calculate shoulder and hip angles
            shoulder_angle = self._calculate_angle(left_shoulder, right_shoulder)
            hip_angle = self._calculate_angle(left_hip, right_hip)
            
            # Determine orientation based on angles
            if 70 <= shoulder_angle <= 110 and 70 <= hip_angle <= 110:
                result["orientation"] = "horizontal"
                result["score"] = 1.0
                result["feedback"] = "Good horizontal dive orientation"
            elif 40 <= shoulder_angle <= 140 and 40 <= hip_angle <= 140:
                result["orientation"] = "acceptable"
                result["score"] = 0.7
                result["feedback"] = "Acceptable dive orientation"
            else:
                result["orientation"] = "poor"
                result["score"] = 0.3
                result["feedback"] = "Poor dive orientation, keep body more horizontal"
                
        except Exception as e:
            result["feedback"] = f"Error analyzing dive orientation: {str(e)}"
            
        return result
    
    def _analyze_dive_quality(self, dive_frames: List[Dict]) -> Dict:
        """Analyze the quality of the dive."""
        result = {"quality": "unknown", "score": 0.0, "feedback": ""}
        
        try:
            if not dive_frames:
                result["feedback"] = "No dive frames provided"
                return result
            
            # Get the first and last frames of the dive
            first_frame = dive_frames[0]
            last_frame = dive_frames[-1]
            
            first_keypoints = first_frame.get("key_points", {})
            last_keypoints = last_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in first_keypoints for k in ["left_hip", "right_hip"]) or \
               not all(k in last_keypoints for k in ["left_hip", "right_hip"]):
                result["feedback"] = "Missing keypoints for dive quality analysis"
                return result
            
            # Calculate hip positions
            first_hip_center = {
                "x": (first_keypoints["left_hip"]["x"] + first_keypoints["right_hip"]["x"]) / 2,
                "y": (first_keypoints["left_hip"]["y"] + first_keypoints["right_hip"]["y"]) / 2
            }
            
            last_hip_center = {
                "x": (last_keypoints["left_hip"]["x"] + last_keypoints["right_hip"]["x"]) / 2,
                "y": (last_keypoints["left_hip"]["y"] + last_keypoints["right_hip"]["y"]) / 2
            }
            
            # Calculate distance traveled
            distance = np.sqrt(
                (last_hip_center["x"] - first_hip_center["x"])**2 + 
                (last_hip_center["y"] - first_hip_center["y"])**2
            )
            
            # Calculate dive smoothness
            smoothness = self._calculate_dive_smoothness(dive_frames)
            
            # Determine quality based on distance and smoothness
            if distance > 0.3 and smoothness > 0.8:
                result["quality"] = "excellent"
                result["score"] = 1.0
                result["feedback"] = "Excellent dive with good extension and smooth motion"
            elif distance > 0.2 and smoothness > 0.6:
                result["quality"] = "good"
                result["score"] = 0.7
                result["feedback"] = "Good dive with decent extension"
            else:
                result["quality"] = "poor"
                result["score"] = 0.3
                result["feedback"] = "Poor dive with limited extension"
                
        except Exception as e:
            result["feedback"] = f"Error analyzing dive quality: {str(e)}"
            
        return result
    
    def _find_throw_frames(self, frames_data: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
        """Find the start and end frames of a throw."""
        try:
            throw_start = None
            throw_end = None
            
            # Look for significant arm movement
            for i in range(1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["right_wrist", "right_elbow"]) or \
                   not all(k in curr_keypoints for k in ["right_wrist", "right_elbow"]):
                    continue
                
                # Calculate wrist movement
                prev_right_wrist = prev_keypoints["right_wrist"]
                curr_right_wrist = curr_keypoints["right_wrist"]
                
                # Calculate distance moved
                distance = np.sqrt(
                    (curr_right_wrist["x"] - prev_right_wrist["x"])**2 + 
                    (curr_right_wrist["y"] - prev_right_wrist["y"])**2
                )
                
                # If wrist movement is significant, it might be a throw
                if distance > self.wrist_extension_threshold:
                    if throw_start is None:
                        throw_start = i - 1
                    throw_end = i
            
            return throw_start, throw_end
            
        except Exception as e:
            print(f"Error finding throw frames: {str(e)}")
            return None, None
    
    def _analyze_shoulder_rotation(self, throw_frames: List[Dict]) -> float:
        """Analyze the shoulder rotation during the throw."""
        try:
            if not throw_frames:
                return 0.0
            
            # Get the first and last frames of the throw
            first_frame = throw_frames[0]
            last_frame = throw_frames[-1]
            
            first_keypoints = first_frame.get("key_points", {})
            last_keypoints = last_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in first_keypoints for k in ["left_shoulder", "right_shoulder"]) or \
               not all(k in last_keypoints for k in ["left_shoulder", "right_shoulder"]):
                return 0.0
            
            # Calculate shoulder angles
            first_shoulder_angle = self._calculate_angle(
                first_keypoints["left_shoulder"], 
                first_keypoints["right_shoulder"]
            )
            
            last_shoulder_angle = self._calculate_angle(
                last_keypoints["left_shoulder"], 
                last_keypoints["right_shoulder"]
            )
            
            # Calculate rotation
            rotation = abs(last_shoulder_angle - first_shoulder_angle)
            
            return rotation
            
        except Exception as e:
            print(f"Error analyzing shoulder rotation: {str(e)}")
            return 0.0
    
    def _analyze_wrist_extension(self, throw_frames: List[Dict]) -> float:
        """Analyze the wrist extension during the throw."""
        try:
            if not throw_frames:
                return 0.0
            
            # Get the first and last frames of the throw
            first_frame = throw_frames[0]
            last_frame = throw_frames[-1]
            
            first_keypoints = first_frame.get("key_points", {})
            last_keypoints = last_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in first_keypoints for k in ["right_wrist", "right_elbow"]) or \
               not all(k in last_keypoints for k in ["right_wrist", "right_elbow"]):
                return 0.0
            
            # Calculate wrist distances from elbow
            first_wrist_distance = self._calculate_distance(
                first_keypoints["right_wrist"], 
                first_keypoints["right_elbow"]
            )
            
            last_wrist_distance = self._calculate_distance(
                last_keypoints["right_wrist"], 
                last_keypoints["right_elbow"]
            )
            
            # Calculate extension
            extension = last_wrist_distance - first_wrist_distance
            
            return extension
            
        except Exception as e:
            print(f"Error analyzing wrist extension: {str(e)}")
            return 0.0
    
    def _detect_arm_lag(self, throw_frames: List[Dict]) -> bool:
        """Detect if there is arm lag during the throw."""
        try:
            if len(throw_frames) < 3:
                return False
            
            # Get three frames during the throw
            first_frame = throw_frames[0]
            mid_frame = throw_frames[len(throw_frames) // 2]
            last_frame = throw_frames[-1]
            
            first_keypoints = first_frame.get("key_points", {})
            mid_keypoints = mid_frame.get("key_points", {})
            last_keypoints = last_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in first_keypoints for k in ["right_shoulder", "right_elbow", "right_wrist"]) or \
               not all(k in mid_keypoints for k in ["right_shoulder", "right_elbow", "right_wrist"]) or \
               not all(k in last_keypoints for k in ["right_shoulder", "right_elbow", "right_wrist"]):
                return False
            
            # Calculate arm angles
            first_arm_angle = self._calculate_arm_angle(
                first_keypoints["right_shoulder"], 
                first_keypoints["right_elbow"], 
                first_keypoints["right_wrist"]
            )
            
            mid_arm_angle = self._calculate_arm_angle(
                mid_keypoints["right_shoulder"], 
                mid_keypoints["right_elbow"], 
                mid_keypoints["right_wrist"]
            )
            
            last_arm_angle = self._calculate_arm_angle(
                last_keypoints["right_shoulder"], 
                last_keypoints["right_elbow"], 
                last_keypoints["right_wrist"]
            )
            
            # Detect arm lag (if the arm angle decreases then increases)
            arm_lag = (mid_arm_angle < first_arm_angle) and (last_arm_angle > mid_arm_angle)
            
            return arm_lag
            
        except Exception as e:
            print(f"Error detecting arm lag: {str(e)}")
            return False
    
    def _detect_follow_through(self, throw_frames: List[Dict]) -> bool:
        """Detect if there is a proper follow-through after the throw."""
        try:
            if len(throw_frames) < 3:
                return False
            
            # Get the last few frames of the throw
            last_frame = throw_frames[-1]
            prev_frame = throw_frames[-2]
            
            last_keypoints = last_frame.get("key_points", {})
            prev_keypoints = prev_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in last_keypoints for k in ["right_wrist", "right_elbow"]) or \
               not all(k in prev_keypoints for k in ["right_wrist", "right_elbow"]):
                return False
            
            # Calculate wrist positions
            last_wrist = last_keypoints["right_wrist"]
            prev_wrist = prev_keypoints["right_wrist"]
            last_elbow = last_keypoints["right_elbow"]
            
            # Calculate wrist movement
            dx = last_wrist["x"] - prev_wrist["x"]
            dy = last_wrist["y"] - prev_wrist["y"]
            
            # Calculate distance from wrist to elbow
            wrist_elbow_distance = self._calculate_distance(last_wrist, last_elbow)
            
            # Detect follow-through (if the wrist continues to move after release and is extended)
            follow_through = (abs(dx) > 0.01 or abs(dy) > 0.01) and (wrist_elbow_distance > 0.2)
            
            return follow_through
            
        except Exception as e:
            print(f"Error detecting follow-through: {str(e)}")
            return False
    
    def _find_ball_visible_frame(self, frames_data: List[Dict]) -> Optional[int]:
        """Find the frame where the ball becomes visible (simulated)."""
        try:
            # In a real implementation, this would use ball tracking data
            # For now, we'll simulate it by finding the frame with maximum movement
            max_movement = 0
            ball_visible_frame = None
            
            for i in range(1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_hip", "right_hip"]) or \
                   not all(k in curr_keypoints for k in ["left_hip", "right_hip"]):
                    continue
                
                # Calculate hip movement
                prev_hip_center = {
                    "x": (prev_keypoints["left_hip"]["x"] + prev_keypoints["right_hip"]["x"]) / 2,
                    "y": (prev_keypoints["left_hip"]["y"] + prev_keypoints["right_hip"]["y"]) / 2
                }
                
                curr_hip_center = {
                    "x": (curr_keypoints["left_hip"]["x"] + curr_keypoints["right_hip"]["x"]) / 2,
                    "y": (curr_keypoints["left_hip"]["y"] + curr_keypoints["right_hip"]["y"]) / 2
                }
                
                # Calculate movement
                movement = np.sqrt(
                    (curr_hip_center["x"] - prev_hip_center["x"])**2 + 
                    (curr_hip_center["y"] - prev_hip_center["y"])**2
                )
                
                if movement > max_movement:
                    max_movement = movement
                    ball_visible_frame = i
            
            return ball_visible_frame
            
        except Exception as e:
            print(f"Error finding ball visible frame: {str(e)}")
            return None
    
    def _find_movement_start_frame(self, frames_data: List[Dict], ball_visible_frame: int) -> Optional[int]:
        """Find the frame where the fielder starts moving after the ball becomes visible."""
        try:
            if ball_visible_frame is None or ball_visible_frame >= len(frames_data):
                return None
            
            # Look for movement after the ball becomes visible
            for i in range(ball_visible_frame + 1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_hip", "right_hip"]) or \
                   not all(k in curr_keypoints for k in ["left_hip", "right_hip"]):
                    continue
                
                # Calculate hip movement
                prev_hip_center = {
                    "x": (prev_keypoints["left_hip"]["x"] + prev_keypoints["right_hip"]["x"]) / 2,
                    "y": (prev_keypoints["left_hip"]["y"] + prev_keypoints["right_hip"]["y"]) / 2
                }
                
                curr_hip_center = {
                    "x": (curr_keypoints["left_hip"]["x"] + curr_keypoints["right_hip"]["x"]) / 2,
                    "y": (curr_keypoints["left_hip"]["y"] + curr_keypoints["right_hip"]["y"]) / 2
                }
                
                # Calculate movement
                movement = np.sqrt(
                    (curr_hip_center["x"] - prev_hip_center["x"])**2 + 
                    (curr_hip_center["y"] - prev_hip_center["y"])**2
                )
                
                # If movement is significant, we've found the movement start
                if movement > 0.05:
                    return i
            
            return None
            
        except Exception as e:
            print(f"Error finding movement start frame: {str(e)}")
            return None
    
    def _calculate_angle(self, point1: Dict, point2: Dict) -> float:
        """Calculate the angle of the line between two points."""
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
    
    def _calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate the distance between two points."""
        try:
            dx = point2["x"] - point1["x"]
            dy = point2["y"] - point1["y"]
            
            distance = np.sqrt(dx**2 + dy**2)
            
            return distance
            
        except Exception as e:
            print(f"Error calculating distance: {str(e)}")
            return 0.0
    
    def _calculate_arm_angle(self, shoulder: Dict, elbow: Dict, wrist: Dict) -> float:
        """Calculate the angle of the arm."""
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
    
    def _calculate_dive_smoothness(self, dive_frames: List[Dict]) -> float:
        """Calculate the smoothness of the dive."""
        try:
            if len(dive_frames) < 3:
                return 0.0
            
            # Calculate hip center positions
            hip_positions = []
            for frame in dive_frames:
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if not all(k in keypoints for k in ["left_hip", "right_hip"]):
                    continue
                
                hip_center = {
                    "x": (keypoints["left_hip"]["x"] + keypoints["right_hip"]["x"]) / 2,
                    "y": (keypoints["left_hip"]["y"] + keypoints["right_hip"]["y"]) / 2
                }
                
                hip_positions.append(hip_center)
            
            if len(hip_positions) < 3:
                return 0.0
            
            # Calculate accelerations
            accelerations = []
            for i in range(1, len(hip_positions) - 1):
                # Calculate velocity
                v1x = hip_positions[i]["x"] - hip_positions[i-1]["x"]
                v1y = hip_positions[i]["y"] - hip_positions[i-1]["y"]
                v2x = hip_positions[i+1]["x"] - hip_positions[i]["x"]
                v2y = hip_positions[i+1]["y"] - hip_positions[i]["y"]
                
                # Calculate acceleration
                ax = v2x - v1x
                ay = v2y - v1y
                
                # Calculate acceleration magnitude
                accel_mag = np.sqrt(ax**2 + ay**2)
                accelerations.append(accel_mag)
            
            # Calculate smoothness (inverse of acceleration variance)
            if not accelerations:
                return 0.0
                
            accel_variance = np.var(accelerations)
            smoothness = 1.0 / (1.0 + accel_variance)
            
            return smoothness
            
        except Exception as e:
            print(f"Error calculating dive smoothness: {str(e)}")
            return 0.0

    def _analyze_throwing_technique_correctness(self, frames_data: List[Dict]) -> Dict:
        """Analyze throwing technique correctness with biomechanical evaluation."""
        result = {
            "technique_correctness": "unknown",
            "shoulder_rotation": 0.0,
            "elbow_position": 0.0,
            "wrist_snap": 0.0,
            "follow_through": False,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find the throw frame (frame with maximum arm speed)
            throw_frame_idx = self._find_throw_frame(frames_data)
            if throw_frame_idx is None:
                result["feedback"] = "Could not determine throw frame"
                return result
            
            # Get frames for analysis
            start_idx = max(0, throw_frame_idx - 5)
            end_idx = min(len(frames_data) - 1, throw_frame_idx + 10)
            relevant_frames = frames_data[start_idx:end_idx + 1]
            
            # Analyze shoulder rotation
            shoulder_rotations = []
            for frame in relevant_frames:
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                    continue
                
                # Calculate shoulder-hip rotation
                left_shoulder = keypoints["left_shoulder"]
                right_shoulder = keypoints["right_shoulder"]
                left_hip = keypoints["left_hip"]
                right_hip = keypoints["right_hip"]
                
                shoulder_vector = np.array([
                    right_shoulder["x"] - left_shoulder["x"],
                    right_shoulder["y"] - left_shoulder["y"]
                ])
                
                hip_vector = np.array([
                    right_hip["x"] - left_hip["x"],
                    right_hip["y"] - left_hip["y"]
                ])
                
                # Normalize vectors
                shoulder_norm = np.linalg.norm(shoulder_vector)
                hip_norm = np.linalg.norm(hip_vector)
                
                if shoulder_norm > 0 and hip_norm > 0:
                    shoulder_vector = shoulder_vector / shoulder_norm
                    hip_vector = hip_vector / hip_norm
                    
                    # Calculate angle between vectors
                    dot_product = np.dot(shoulder_vector, hip_vector)
                    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
                    shoulder_rotation = np.degrees(angle_rad)
                    shoulder_rotations.append(shoulder_rotation)
            
            if shoulder_rotations:
                result["shoulder_rotation"] = max(shoulder_rotations)
                
                # Evaluate shoulder rotation (ideal is > 45 degrees)
                if result["shoulder_rotation"] > 45:
                    shoulder_score = 1.0
                    shoulder_feedback = "✅ Good shoulder rotation"
                else:
                    shoulder_score = result["shoulder_rotation"] / 45
                    shoulder_feedback = "❌ Insufficient shoulder rotation"
            else:
                shoulder_score = 0.0
                shoulder_feedback = "Could not determine shoulder rotation"
            
            # Analyze elbow position
            throw_frame = relevant_frames[throw_frame_idx - start_idx]
            keypoints = throw_frame.get("key_points", {})
            
            if all(k in keypoints for k in ["right_shoulder", "right_elbow", "right_wrist"]):
                right_shoulder = keypoints["right_shoulder"]
                right_elbow = keypoints["right_elbow"]
                right_wrist = keypoints["right_wrist"]
                
                # Calculate elbow position (should be above shoulder for good technique)
                elbow_height = right_shoulder["y"] - right_elbow["y"]
                
                if elbow_height > 0.05:  # Elbow above shoulder
                    result["elbow_position"] = elbow_height
                    elbow_score = 1.0
                    elbow_feedback = "✅ Good elbow position"
                else:
                    result["elbow_position"] = elbow_height
                    elbow_score = elbow_height / 0.05
                    elbow_feedback = "❌ Elbow too low"
            else:
                elbow_score = 0.0
                elbow_feedback = "Could not determine elbow position"
            
            # Analyze wrist snap
            wrist_positions = []
            for frame in relevant_frames:
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if "right_wrist" in keypoints:
                    wrist_positions.append(keypoints["right_wrist"])
            
            if len(wrist_positions) > 1:
                # Calculate wrist velocities
                velocities = []
                for i in range(1, len(wrist_positions)):
                    dx = wrist_positions[i]["x"] - wrist_positions[i-1]["x"]
                    dy = wrist_positions[i]["y"] - wrist_positions[i-1]["y"]
                    velocity = np.sqrt(dx**2 + dy**2)
                    velocities.append(velocity)
                
                # Calculate acceleration (snap)
                accelerations = []
                for i in range(1, len(velocities)):
                    acceleration = velocities[i] - velocities[i-1]
                    accelerations.append(acceleration)
                
                if accelerations:
                    max_acceleration = max(accelerations)
                    result["wrist_snap"] = max_acceleration
                    
                    # Evaluate wrist snap
                    if max_acceleration > 0.1:
                        wrist_score = 1.0
                        wrist_feedback = "✅ Good wrist snap"
                    else:
                        wrist_score = max_acceleration / 0.1
                        wrist_feedback = "❌ Weak wrist snap"
                else:
                    wrist_score = 0.0
                    wrist_feedback = "Could not determine wrist snap"
            else:
                wrist_score = 0.0
                wrist_feedback = "Could not determine wrist snap"
            
            # Analyze follow-through
            if len(relevant_frames) > 5:
                # Check if arm continues to move after release
                post_release_frames = relevant_frames[5:]
                arm_movement = False
                
                for i in range(1, len(post_release_frames)):
                    if not post_release_frames[i-1].get("pose_detected") or \
                       not post_release_frames[i].get("pose_detected"):
                        continue
                    
                    prev_keypoints = post_release_frames[i-1].get("key_points", {})
                    curr_keypoints = post_release_frames[i].get("key_points", {})
                    
                    if "right_wrist" in prev_keypoints and "right_wrist" in curr_keypoints:
                        dx = curr_keypoints["right_wrist"]["x"] - prev_keypoints["right_wrist"]["x"]
                        dy = curr_keypoints["right_wrist"]["y"] - prev_keypoints["right_wrist"]["y"]
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        if distance > 0.05:  # Significant movement
                            arm_movement = True
                            break
                
                result["follow_through"] = arm_movement
                
                if arm_movement:
                    follow_through_score = 1.0
                    follow_through_feedback = "✅ Good follow-through"
                else:
                    follow_through_score = 0.0
                    follow_through_feedback = "❌ Incomplete follow-through"
            else:
                follow_through_score = 0.0
                follow_through_feedback = "Could not determine follow-through"
            
            # Calculate overall score
            result["score"] = (shoulder_score + elbow_score + wrist_score + follow_through_score) / 4
            
            # Determine overall technique correctness
            if result["score"] > 0.8:
                result["technique_correctness"] = "good"
            elif result["score"] > 0.6:
                result["technique_correctness"] = "acceptable"
            else:
                result["technique_correctness"] = "poor"
            
            # Generate comprehensive feedback
            feedback_parts = [shoulder_feedback, elbow_feedback, wrist_feedback, follow_through_feedback]
            result["feedback"] = ". ".join(feedback_parts)
            
        except Exception as e:
            result["feedback"] = f"Error analyzing throwing technique correctness: {str(e)}"
            
        return result

