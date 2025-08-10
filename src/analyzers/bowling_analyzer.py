# src/analyzers/bowling_analyzer.py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class BowlingAnalyzer:
    def __init__(self):
        # Initialize thresholds and parameters
        self.min_confidence = 0.5
        self.run_up_phase_threshold = 0.2  # Normalized movement threshold
        self.load_up_phase_threshold = 0.3  # Threshold for detecting load-up
        self.release_angle_thresholds = {
            "fast": (70, 90),
            "medium": (50, 70),
            "spin": (30, 50)
        }
        self.wrist_flick_threshold = 0.15  # Threshold for detecting wrist flick
        
    def analyze(self, frames_data: List[Dict]) -> Dict:
        """
        Enhanced bowling analysis with phase segmentation and release dynamics.
        
        Args:
            frames_data: List of dictionaries containing frame data with pose keypoints
            
        Returns:
            Dictionary containing bowling analysis results
        """
        results = {
            "phases": {
                "run_up": {"detected": False, "start_frame": -1, "end_frame": -1, "score": 0.0, "feedback": ""},
                "load_up": {"detected": False, "start_frame": -1, "end_frame": -1, "score": 0.0, "feedback": ""},
                "delivery": {"detected": False, "start_frame": -1, "end_frame": -1, "score": 0.0, "feedback": ""},
                "follow_through": {"detected": False, "start_frame": -1, "end_frame": -1, "score": 0.0, "feedback": ""}
            },
            "release_dynamics": {
                "arm_angle": 0.0,
                "wrist_flick": {"detected": False, "score": 0.0, "feedback": ""},
                "shoulder_hip_torque": 0.0,
                "score": 0.0,
                "feedback": ""
            },
            "run_up_consistency": {"score": 0.0, "feedback": ""},
            "front_foot_landing": {"position": "unknown", "score": 0.0, "feedback": ""},
            "scores": {
                "run_up_score": 0.0,
                "load_up_score": 0.0,
                "delivery_score": 0.0,
                "follow_through_score": 0.0,
                "overall_score": 0.0
            },
            "corrections": []
        }
        
        if not frames_data:
            results["error"] = "No frame data provided"
            return results
            
        try:
            # Segment the bowling action into phases
            self._segment_phases(frames_data, results)
            
            # Analyze release dynamics
            release_results = self._analyze_release_dynamics(frames_data, results)
            results["release_dynamics"] = release_results
            
            # Analyze run-up consistency
            run_up_results = self._analyze_run_up_consistency(frames_data, results)
            results["run_up_consistency"] = run_up_results
            
            # Analyze front foot landing
            foot_landing_results = self._analyze_front_foot_landing(frames_data, results)
            results["front_foot_landing"] = foot_landing_results
            
            # Calculate overall scores
            self._calculate_overall_scores(results)
            
            # Generate corrections
            results["corrections"] = self._generate_corrections(results)
            
        except Exception as e:
            results["error"] = str(e)
            
        return results
    
    def _segment_phases(self, frames_data: List[Dict], results: Dict) -> None:
        """Segment the bowling action into run-up, load-up, delivery, and follow-through phases."""
        try:
            # Find the delivery frame (frame with maximum arm speed)
            delivery_frame_idx = self._find_delivery_frame(frames_data)
            if delivery_frame_idx is None:
                results["phases"]["delivery"]["feedback"] = "Could not determine delivery frame"
                return
                
            # Set delivery phase
            results["phases"]["delivery"]["detected"] = True
            results["phases"]["delivery"]["start_frame"] = delivery_frame_idx - 5
            results["phases"]["delivery"]["end_frame"] = delivery_frame_idx + 5
            results["phases"]["delivery"]["score"] = 1.0
            results["phases"]["delivery"]["feedback"] = "Delivery phase detected"
            
            # Find load-up phase (before delivery)
            load_up_start = self._find_load_up_start(frames_data, delivery_frame_idx)
            if load_up_start is not None:
                results["phases"]["load_up"]["detected"] = True
                results["phases"]["load_up"]["start_frame"] = load_up_start
                results["phases"]["load_up"]["end_frame"] = delivery_frame_idx - 6
                results["phases"]["load_up"]["score"] = 1.0
                results["phases"]["load_up"]["feedback"] = "Load-up phase detected"
            
            # Find run-up phase (before load-up)
            if load_up_start is not None and load_up_start > 10:
                results["phases"]["run_up"]["detected"] = True
                results["phases"]["run_up"]["start_frame"] = 0
                results["phases"]["run_up"]["end_frame"] = load_up_start - 1
                results["phases"]["run_up"]["score"] = 1.0
                results["phases"]["run_up"]["feedback"] = "Run-up phase detected"
            
            # Find follow-through phase (after delivery)
            follow_through_end = self._find_follow_through_end(frames_data, delivery_frame_idx)
            if follow_through_end is not None:
                results["phases"]["follow_through"]["detected"] = True
                results["phases"]["follow_through"]["start_frame"] = delivery_frame_idx + 6
                results["phases"]["follow_through"]["end_frame"] = follow_through_end
                results["phases"]["follow_through"]["score"] = 1.0
                results["phases"]["follow_through"]["feedback"] = "Follow-through phase detected"
                
        except Exception as e:
            for phase in results["phases"].values():
                phase["feedback"] = f"Error segmenting phases: {str(e)}"
    
    def _analyze_release_dynamics(self, frames_data: List[Dict], results: Dict) -> Dict:
        """Analyze the release dynamics including arm angle, wrist flick, and shoulder-hip torque."""
        release_result = {
            "arm_angle": 0.0,
            "wrist_flick": {"detected": False, "score": 0.0, "feedback": ""},
            "shoulder_hip_torque": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Get delivery frame
            delivery_frame_idx = results["phases"]["delivery"]["start_frame"] + 5
            if delivery_frame_idx >= len(frames_data):
                release_result["feedback"] = "Invalid delivery frame index"
                return release_result
                
            delivery_frame = frames_data[delivery_frame_idx]
            keypoints = delivery_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                               "left_wrist", "right_wrist", "left_hip", "right_hip"]):
                release_result["feedback"] = "Missing keypoints for release dynamics analysis"
                return release_result
            
            # Calculate arm release angle
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            left_elbow = keypoints["left_elbow"]
            right_elbow = keypoints["right_elbow"]
            left_wrist = keypoints["left_wrist"]
            right_wrist = keypoints["right_wrist"]
            
            # Determine bowling arm (assuming right-handed bowler)
            bowling_arm_shoulder = right_shoulder
            bowling_arm_elbow = right_elbow
            bowling_arm_wrist = right_wrist
            
            # Calculate arm angle at release
            arm_angle = self._calculate_arm_angle(bowling_arm_shoulder, bowling_arm_elbow, bowling_arm_wrist)
            release_result["arm_angle"] = arm_angle
            
            # Analyze wrist flick
            wrist_flick_result = self._analyze_wrist_flick(frames_data, delivery_frame_idx)
            release_result["wrist_flick"] = wrist_flick_result
            
            # Calculate shoulder-hip torque
            left_hip = keypoints["left_hip"]
            right_hip = keypoints["right_hip"]
            shoulder_hip_torque = self._calculate_shoulder_hip_torque(
                left_shoulder, right_shoulder, left_hip, right_hip
            )
            release_result["shoulder_hip_torque"] = shoulder_hip_torque
            
            # Calculate overall release dynamics score
            arm_angle_score = self._evaluate_arm_angle(arm_angle)
            wrist_flick_score = wrist_flick_result["score"]
            torque_score = min(shoulder_hip_torque / 100, 1.0)  # Normalize torque
            
            release_result["score"] = (
                arm_angle_score * 0.4 +
                wrist_flick_score * 0.3 +
                torque_score * 0.3
            )
            
            # Generate feedback
            feedback_parts = []
            if arm_angle_score < 0.7:
                feedback_parts.append(f"Low release angle ({arm_angle:.1f}°)")
            if wrist_flick_score < 0.7:
                feedback_parts.append("Weak wrist flick")
            if torque_score < 0.7:
                feedback_parts.append("Poor shoulder-hip torque")
                
            if feedback_parts:
                release_result["feedback"] = ", ".join(feedback_parts)
            else:
                release_result["feedback"] = "Good release dynamics"
                
        except Exception as e:
            release_result["feedback"] = f"Error analyzing release dynamics: {str(e)}"
            
        return release_result
    
    def _analyze_run_up_consistency(self, frames_data: List[Dict], results: Dict) -> Dict:
        """Analyze the consistency of the run-up."""
        run_up_result = {"score": 0.0, "feedback": ""}
        
        try:
            # Get run-up frames
            run_up_start = results["phases"]["run_up"]["start_frame"]
            run_up_end = results["phases"]["run_up"]["end_frame"]
            
            if not results["phases"]["run_up"]["detected"] or run_up_start == -1 or run_up_end == -1:
                run_up_result["feedback"] = "Run-up phase not detected"
                return run_up_result
            
            # Extract foot positions during run-up
            left_foot_positions = []
            right_foot_positions = []
            
            for i in range(run_up_start, run_up_end + 1):
                frame = frames_data[i]
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if "left_ankle" in keypoints:
                    left_foot_positions.append(keypoints["left_ankle"])
                if "right_ankle" in keypoints:
                    right_foot_positions.append(keypoints["right_ankle"])
            
            if not left_foot_positions or not right_foot_positions:
                run_up_result["feedback"] = "Insufficient foot position data for run-up analysis"
                return run_up_result
            
            # Calculate stride consistency
            left_stride_lengths = self._calculate_stride_lengths(left_foot_positions)
            right_stride_lengths = self._calculate_stride_lengths(right_foot_positions)
            
            # Calculate consistency scores
            left_consistency = self._calculate_consistency(left_stride_lengths)
            right_consistency = self._calculate_consistency(right_stride_lengths)
            
            # Overall consistency is the average of left and right
            overall_consistency = (left_consistency + right_consistency) / 2
            run_up_result["score"] = overall_consistency
            
            # Generate feedback
            if overall_consistency > 0.8:
                run_up_result["feedback"] = "Very consistent run-up"
            elif overall_consistency > 0.6:
                run_up_result["feedback"] = "Fairly consistent run-up"
            else:
                run_up_result["feedback"] = "Inconsistent run-up, work on rhythm"
                
        except Exception as e:
            run_up_result["feedback"] = f"Error analyzing run-up consistency: {str(e)}"
            
        return run_up_result
    
    def _analyze_front_foot_landing(self, frames_data: List[Dict], results: Dict) -> Dict:
        """Analyze the front foot landing position."""
        foot_landing_result = {"position": "unknown", "score": 0.0, "feedback": ""}
        
        try:
            # Get delivery frame
            delivery_frame_idx = results["phases"]["delivery"]["start_frame"] + 5
            if delivery_frame_idx >= len(frames_data):
                foot_landing_result["feedback"] = "Invalid delivery frame index"
                return foot_landing_result
                
            delivery_frame = frames_data[delivery_frame_idx]
            keypoints = delivery_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in keypoints for k in ["left_ankle", "right_ankle", "left_hip", "right_hip"]):
                foot_landing_result["feedback"] = "Missing keypoints for foot landing analysis"
                return foot_landing_result
            
            # Extract keypoints
            left_ankle = keypoints["left_ankle"]
            right_ankle = keypoints["right_ankle"]
            left_hip = keypoints["left_hip"]
            right_hip = keypoints["right_hip"]
            
            # Determine which foot is forward (assuming right-handed bowler)
            if left_ankle["y"] < right_ankle["y"]:
                front_foot = left_ankle
                back_foot = right_ankle
                front_foot_label = "left"
            else:
                front_foot = right_ankle
                back_foot = left_ankle
                front_foot_label = "right"
            
            # Calculate hip center
            hip_center = {
                "x": (left_hip["x"] + right_hip["x"]) / 2,
                "y": (left_hip["y"] + right_hip["y"]) / 2
            }
            
            # Calculate distance between front foot and hip center
            foot_hip_distance = np.sqrt(
                (front_foot["x"] - hip_center["x"])**2 + 
                (front_foot["y"] - hip_center["y"])**2
            )
            
            # Calculate alignment of front foot with hip center
            alignment_score = self._calculate_foot_alignment(front_foot, hip_center)
            
            # Determine landing position quality
            if 0.8 <= alignment_score <= 1.0:
                foot_landing_result["position"] = "straight"
                foot_landing_result["score"] = 1.0
                foot_landing_result["feedback"] = f"Good {front_foot_label} foot landing, straight alignment"
            elif alignment_score < 0.8:
                foot_landing_result["position"] = "wide"
                foot_landing_result["score"] = alignment_score
                foot_landing_result["feedback"] = f"{front_foot_label} foot landing too wide"
            else:
                foot_landing_result["position"] = "narrow"
                foot_landing_result["score"] = 2.0 - alignment_score
                foot_landing_result["feedback"] = f"{front_foot_label} foot landing too narrow"
                
        except Exception as e:
            foot_landing_result["feedback"] = f"Error analyzing front foot landing: {str(e)}"
            
        return foot_landing_result
    
    def _calculate_overall_scores(self, results: Dict) -> None:
        """Calculate overall scores from individual analysis components."""
        try:
            # Get phase scores
            run_up_score = results["phases"]["run_up"]["score"] if results["phases"]["run_up"]["detected"] else 0.5
            load_up_score = results["phases"]["load_up"]["score"] if results["phases"]["load_up"]["detected"] else 0.5
            delivery_score = results["phases"]["delivery"]["score"] if results["phases"]["delivery"]["detected"] else 0.5
            follow_through_score = results["phases"]["follow_through"]["score"] if results["phases"]["follow_through"]["detected"] else 0.5
            
            # Get other scores
            release_dynamics_score = results["release_dynamics"]["score"]
            run_up_consistency_score = results["run_up_consistency"]["score"]
            front_foot_landing_score = results["front_foot_landing"]["score"]
            
            # Calculate weighted average
            overall_score = (
                run_up_score * 0.1 +
                load_up_score * 0.2 +
                delivery_score * 0.3 +
                follow_through_score * 0.1 +
                release_dynamics_score * 0.2 +
                run_up_consistency_score * 0.05 +
                front_foot_landing_score * 0.05
            )
            
            # Update scores dictionary
            results["scores"]["run_up_score"] = run_up_score
            results["scores"]["load_up_score"] = load_up_score
            results["scores"]["delivery_score"] = delivery_score
            results["scores"]["follow_through_score"] = follow_through_score
            results["scores"]["overall_score"] = overall_score
            
        except Exception as e:
            print(f"Error calculating overall scores: {str(e)}")
    
    def _generate_corrections(self, results: Dict) -> List[str]:
        """Generate corrective feedback based on analysis results."""
        corrections = []
        
        try:
            # Phase detection corrections
            if not results["phases"]["run_up"]["detected"]:
                corrections.append("Develop a consistent run-up rhythm")
            
            if not results["phases"]["load_up"]["detected"]:
                corrections.append("Work on your load-up technique for better power transfer")
            
            if not results["phases"]["follow_through"]["detected"]:
                corrections.append("Complete your follow-through for better control")
            
            # Release dynamics corrections
            if results["release_dynamics"]["score"] < 0.7:
                if results["release_dynamics"]["arm_angle"] < 50:
                    corrections.append("Increase your arm release angle for better pace")
                elif results["release_dynamics"]["arm_angle"] > 90:
                    corrections.append("Lower your arm release angle for better control")
                
                if not results["release_dynamics"]["wrist_flick"]["detected"]:
                    corrections.append("Work on your wrist flick for better ball movement")
                
                if results["release_dynamics"]["shoulder_hip_torque"] < 50:
                    corrections.append("Improve shoulder-hip separation for more power")
            
            # Run-up consistency corrections
            if results["run_up_consistency"]["score"] < 0.7:
                corrections.append("Practice run-up consistency for better approach to the crease")
            
            # Front foot landing corrections
            if results["front_foot_landing"]["score"] < 0.7:
                if results["front_foot_landing"]["position"] == "wide":
                    corrections.append("Land your front foot straighter for better direction")
                elif results["front_foot_landing"]["position"] == "narrow":
                    corrections.append("Widen your front foot landing for better balance")
            
            # If no specific corrections, provide general encouragement
            if not corrections:
                corrections.append("Good bowling technique overall - continue practicing!")
                
        except Exception as e:
            corrections.append(f"Error generating corrections: {str(e)}")
            
        return corrections
    
    def _find_delivery_frame(self, frames_data: List[Dict]) -> Optional[int]:
        """Find the frame where the ball is delivered."""
        try:
            # Look for the frame with maximum arm speed
            max_speed = 0
            delivery_frame_idx = None
            
            for i in range(1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["right_wrist", "right_elbow"]) or \
                   not all(k in curr_keypoints for k in ["right_wrist", "right_elbow"]):
                    continue
                
                # Calculate arm speed (assuming right-handed bowler)
                prev_right_wrist = prev_keypoints["right_wrist"]
                prev_right_elbow = prev_keypoints["right_elbow"]
                curr_right_wrist = curr_keypoints["right_wrist"]
                curr_right_elbow = curr_keypoints["right_elbow"]
                
                # Calculate distance moved by wrist
                distance = np.sqrt(
                    (curr_right_wrist["x"] - prev_right_wrist["x"])**2 + 
                    (curr_right_wrist["y"] - prev_right_wrist["y"])**2
                )
                
                if distance > max_speed:
                    max_speed = distance
                    delivery_frame_idx = i
            
            return delivery_frame_idx
            
        except Exception as e:
            print(f"Error finding delivery frame: {str(e)}")
            return None
    
    def _find_load_up_start(self, frames_data: List[Dict], delivery_frame_idx: int) -> Optional[int]:
        """Find the frame where the load-up phase starts."""
        try:
            # Look backwards from the delivery frame
            for i in range(delivery_frame_idx - 1, 0, -1):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["right_shoulder", "right_hip"]) or \
                   not all(k in curr_keypoints for k in ["right_shoulder", "right_hip"]):
                    continue
                
                # Calculate shoulder-hip separation
                prev_shoulder_hip_sep = self._calculate_shoulder_hip_separation(
                    prev_keypoints["right_shoulder"], prev_keypoints["right_hip"]
                )
                curr_shoulder_hip_sep = self._calculate_shoulder_hip_separation(
                    curr_keypoints["right_shoulder"], curr_keypoints["right_hip"]
                )
                
                # If there's a significant increase in separation, we've found the load-up start
                if curr_shoulder_hip_sep - prev_shoulder_hip_sep > self.load_up_phase_threshold:
                    return i
            
            # If we didn't find a clear start, estimate based on distance from delivery
            return max(0, delivery_frame_idx - 15)
            
        except Exception as e:
            print(f"Error finding load-up start: {str(e)}")
            return None
    
    def _find_follow_through_end(self, frames_data: List[Dict], delivery_frame_idx: int) -> Optional[int]:
        """Find the frame where the follow-through phase ends."""
        try:
            # Look forwards from the delivery frame
            for i in range(delivery_frame_idx + 1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["right_wrist"]) or \
                   not all(k in curr_keypoints for k in ["right_wrist"]):
                    continue
                
                # Calculate wrist movement
                prev_right_wrist = prev_keypoints["right_wrist"]
                curr_right_wrist = curr_keypoints["right_wrist"]
                
                # Calculate distance moved
                distance = np.sqrt(
                    (curr_right_wrist["x"] - prev_right_wrist["x"])**2 + 
                    (curr_right_wrist["y"] - prev_right_wrist["y"])**2
                )
                
                # If the wrist movement drops below a threshold, we've found the end
                if distance < 2.0:  # Threshold for minimal movement
                    return i
            
            # If we didn't find a clear end, return the last frame
            return len(frames_data) - 1
            
        except Exception as e:
            print(f"Error finding follow-through end: {str(e)}")
            return None
    
    def _calculate_arm_angle(self, shoulder: Dict, elbow: Dict, wrist: Dict) -> float:
        """Calculate the angle of the bowling arm at release."""
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
    
    def _analyze_wrist_flick(self, frames_data: List[Dict], delivery_frame_idx: int) -> Dict:
        """Analyze the wrist flick at the point of release."""
        wrist_flick_result = {"detected": False, "score": 0.0, "feedback": ""}
        
        try:
            # Get frames before and after delivery
            start_idx = max(0, delivery_frame_idx - 3)
            end_idx = min(len(frames_data) - 1, delivery_frame_idx + 3)
            relevant_frames = frames_data[start_idx:end_idx + 1]
            
            # Extract wrist positions
            wrist_positions = []
            for frame in relevant_frames:
                if frame.get("pose_detected") and "right_wrist" in frame.get("key_points", {}):
                    wrist_positions.append(frame["key_points"]["right_wrist"])
            
            if len(wrist_positions) < 3:
                wrist_flick_result["feedback"] = "Insufficient wrist position data for flick analysis"
                return wrist_flick_result
            
            # Calculate wrist velocities
            wrist_velocities = []
            for i in range(1, len(wrist_positions)):
                dx = wrist_positions[i]["x"] - wrist_positions[i-1]["x"]
                dy = wrist_positions[i]["y"] - wrist_positions[i-1]["y"]
                velocity = np.sqrt(dx**2 + dy**2)
                wrist_velocities.append(velocity)
            
            # Calculate wrist curvature (change in direction)
            wrist_curvatures = []
            for i in range(1, len(wrist_velocities)):
                curvature = abs(wrist_velocities[i] - wrist_velocities[i-1])
                wrist_curvatures.append(curvature)
            
            # Determine if there's a flick based on velocity and curvature
            max_velocity = max(wrist_velocities) if wrist_velocities else 0
            max_curvature = max(wrist_curvatures) if wrist_curvatures else 0
            
            if max_velocity > self.wrist_flick_threshold and max_curvature > self.wrist_flick_threshold:
                wrist_flick_result["detected"] = True
                wrist_flick_result["score"] = min(1.0, (max_velocity + max_curvature) / 2)
                wrist_flick_result["feedback"] = "Good wrist flick detected"
            else:
                wrist_flick_result["score"] = min(1.0, (max_velocity + max_curvature) / 2)
                wrist_flick_result["feedback"] = "Weak or no wrist flick detected"
                
        except Exception as e:
            wrist_flick_result["feedback"] = f"Error analyzing wrist flick: {str(e)}"
            
        return wrist_flick_result
    
    def _calculate_shoulder_hip_torque(self, left_shoulder: Dict, right_shoulder: Dict, 
                                      left_hip: Dict, right_hip: Dict) -> float:
        """Calculate the torque between shoulders and hips."""
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
            
            # Calculate torque (simplified as the angle difference)
            torque = angle_deg
            
            return torque
            
        except Exception as e:
            print(f"Error calculating shoulder-hip torque: {str(e)}")
            return 0.0
    
    def _evaluate_arm_angle(self, arm_angle: float) -> float:
        """Evaluate the arm angle and return a score."""
        try:
            # Ideal arm angle is between 70 and 90 degrees
            if 70 <= arm_angle <= 90:
                return 1.0
            elif 50 <= arm_angle < 70:
                return 0.5 + (arm_angle - 50) / 40
            elif 90 < arm_angle <= 110:
                return 0.5 + (110 - arm_angle) / 40
            else:
                return 0.3
                
        except Exception as e:
            print(f"Error evaluating arm angle: {str(e)}")
            return 0.0
    
    def _calculate_stride_lengths(self, foot_positions: List[Dict]) -> List[float]:
        """Calculate the lengths of strides during run-up."""
        stride_lengths = []
        
        try:
            for i in range(1, len(foot_positions)):
                dx = foot_positions[i]["x"] - foot_positions[i-1]["x"]
                dy = foot_positions[i]["y"] - foot_positions[i-1]["y"]
                stride_length = np.sqrt(dx**2 + dy**2)
                stride_lengths.append(stride_length)
                
        except Exception as e:
            print(f"Error calculating stride lengths: {str(e)}")
            
        return stride_lengths
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate the consistency of a list of values."""
        try:
            if not values or len(values) < 2:
                return 0.0
                
            # Calculate coefficient of variation (standard deviation / mean)
            mean = np.mean(values)
            std = np.std(values)
            
            if mean == 0:
                return 0.0
                
            cv = std / mean
            
            # Convert to consistency score (lower CV means higher consistency)
            consistency = max(0, 1.0 - cv)
            
            return consistency
            
        except Exception as e:
            print(f"Error calculating consistency: {str(e)}")
            return 0.0
    
    def _calculate_foot_alignment(self, foot: Dict, hip_center: Dict) -> float:
        """Calculate the alignment of the foot with the hip center."""
        try:
            # Calculate vector from hip center to foot
            hip_to_foot = np.array([foot["x"] - hip_center["x"], foot["y"] - hip_center["y"]])
            
            # Calculate the angle of this vector with respect to vertical
            vertical = np.array([0, -1])  # Vertical vector pointing up
            
            # Normalize vectors
            hip_to_foot_norm = np.linalg.norm(hip_to_foot)
            if hip_to_foot_norm == 0:
                return 0.0
                
            hip_to_foot = hip_to_foot / hip_to_foot_norm
            
            # Calculate angle between vectors
            dot_product = np.dot(hip_to_foot, vertical)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            # Convert to degrees
            angle_deg = np.degrees(angle_rad)
            
            # Convert to alignment score (0 degrees is perfect alignment)
            alignment = max(0, 1.0 - angle_deg / 90)
            
            return alignment
            
        except Exception as e:
            print(f"Error calculating foot alignment: {str(e)}")
            return 0.0
    
    def _calculate_shoulder_hip_separation(self, shoulder: Dict, hip: Dict) -> float:
        """Calculate the separation between shoulder and hip."""
        try:
            # Calculate distance between shoulder and hip
            dx = shoulder["x"] - hip["x"]
            dy = shoulder["y"] - hip["y"]
            separation = np.sqrt(dx**2 + dy**2)
            
            return separation
            
        except Exception as e:
            print(f"Error calculating shoulder-hip separation: {str(e)}")
            return 0.0

    def _analyze_jump_height_and_run_up_speed(self, frames_data: List[Dict]) -> Dict:
        """Analyze jump height and run-up speed for bowling."""
        result = {
            "jump_height": 0.0,
            "run_up_speed": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find the delivery frame
            delivery_frame_idx = self._find_delivery_frame(frames_data)
            if delivery_frame_idx is None:
                result["feedback"] = "Could not determine delivery frame"
                return result
            
            # Analyze jump height (using knee lift as proxy)
            knee_heights = []
            for i in range(max(0, delivery_frame_idx - 10), delivery_frame_idx + 5):
                if i >= len(frames_data):
                    continue
                    
                frame = frames_data[i]
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if "right_knee" in keypoints and "right_hip" in keypoints:
                    # Calculate knee height relative to hip
                    knee_height = keypoints["right_hip"]["y"] - keypoints["right_knee"]["y"]
                    knee_heights.append(knee_height)
            
            if knee_heights:
                # Maximum knee lift indicates jump height
                result["jump_height"] = max(knee_heights)
                
                # Evaluate jump height
                if 0.2 <= result["jump_height"] <= 0.4:
                    jump_score = 1.0
                    jump_feedback = "✅ Good jump height"
                elif result["jump_height"] < 0.2:
                    jump_score = result["jump_height"] / 0.2
                    jump_feedback = "❌ Jump too low"
                else:
                    jump_score = max(0, 1.0 - (result["jump_height"] - 0.4) / 0.2)
                    jump_feedback = "❌ Jump too high"
            else:
                jump_score = 0.0
                jump_feedback = "Could not determine jump height"
            
            # Analyze run-up speed
            hip_positions = []
            for i in range(max(0, delivery_frame_idx - 30), delivery_frame_idx):
                if i >= len(frames_data):
                    continue
                    
                frame = frames_data[i]
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if "right_hip" in keypoints:
                    hip_positions.append(keypoints["right_hip"])
            
            if len(hip_positions) > 1:
                # Calculate run-up speed
                total_distance = 0
                for i in range(1, len(hip_positions)):
                    dx = hip_positions[i]["x"] - hip_positions[i-1]["x"]
                    dy = hip_positions[i]["y"] - hip_positions[i-1]["y"]
                    distance = np.sqrt(dx**2 + dy**2)
                    total_distance += distance
                
                # Speed = distance / time (assuming 30 fps)
                result["run_up_speed"] = total_distance / (len(hip_positions) / 30.0)
                
                # Evaluate run-up speed
                if 5.0 <= result["run_up_speed"] <= 8.0:
                    speed_score = 1.0
                    speed_feedback = "✅ Good run-up speed"
                elif result["run_up_speed"] < 5.0:
                    speed_score = result["run_up_speed"] / 5.0
                    speed_feedback = "❌ Run-up too slow"
                else:
                    speed_score = max(0, 1.0 - (result["run_up_speed"] - 8.0) / 3.0)
                    speed_feedback = "❌ Run-up too fast"
            else:
                speed_score = 0.0
                speed_feedback = "Could not determine run-up speed"
            
            # Calculate overall score
            result["score"] = (jump_score + speed_score) / 2
            result["feedback"] = f"{jump_feedback}. {speed_feedback}."
            
        except Exception as e:
            result["feedback"] = f"Error analyzing jump height and run-up speed: {str(e)}"
            
        return result

    def _analyze_arm_shoulder_rotation(self, frames_data: List[Dict]) -> Dict:
        """Analyze arm and shoulder rotation with biomechanical comparison to ideal form."""
        result = {
            "arm_angle": 0.0,
            "shoulder_rotation": 0.0,
            "separation_efficiency": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            # Find the delivery frame
            delivery_frame_idx = self._find_delivery_frame(frames_data)
            if delivery_frame_idx is None:
                result["feedback"] = "Could not determine delivery frame"
                return result
            
            delivery_frame = frames_data[delivery_frame_idx]
            keypoints = delivery_frame.get("key_points", {})
            
            # Check for required keypoints
            if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                                               "left_hip", "right_hip"]):
                result["feedback"] = "Missing keypoints for arm/shoulder rotation analysis"
                return result
            
            # Calculate arm angle at release
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            left_elbow = keypoints["left_elbow"]
            right_elbow = keypoints["right_elbow"]
            
            # Calculate arm angle (for bowling arm)
            arm_vector = np.array([
                right_elbow["x"] - right_shoulder["x"],
                right_elbow["y"] - right_shoulder["y"]
            ])
            
            # Normalize vector
            arm_norm = np.linalg.norm(arm_vector)
            if arm_norm > 0:
                arm_vector = arm_vector / arm_norm
            
            # Calculate arm angle from vertical (0 = straight up, 90 = horizontal)
            vertical = np.array([0, -1])
            arm_angle = np.degrees(np.arccos(np.clip(np.dot(arm_vector, vertical), -1.0, 1.0)))
            result["arm_angle"] = arm_angle
            
            # Calculate shoulder-hip separation
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
                result["shoulder_rotation"] = shoulder_rotation
                
                # Calculate separation efficiency (ideal is 45-60 degrees)
                if 45 <= shoulder_rotation <= 60:
                    separation_efficiency = 1.0
                    separation_feedback = "✅ Ideal shoulder-hip separation"
                elif shoulder_rotation < 45:
                    separation_efficiency = shoulder_rotation / 45
                    separation_feedback = "❌ Insufficient shoulder-hip separation"
                else:
                    separation_efficiency = max(0, 1.0 - (shoulder_rotation - 60) / 30)
                    separation_feedback = "❌ Excessive shoulder-hip separation"
                
                result["separation_efficiency"] = separation_efficiency
            else:
                separation_efficiency = 0.0
                separation_feedback = "Could not calculate separation efficiency"
            
            # Evaluate arm angle (ideal is 70-90 degrees)
            if 70 <= arm_angle <= 90:
                arm_score = 1.0
                arm_feedback = "✅ Ideal arm release angle"
            elif arm_angle < 70:
                arm_score = arm_angle / 70
                arm_feedback = "❌ Arm release angle too low"
            else:
                arm_score = max(0, 1.0 - (arm_angle - 90) / 30)
                arm_feedback = "❌ Arm release angle too high"
            
            # Calculate overall score
            result["score"] = (arm_score + separation_efficiency) / 2
            result["feedback"] = f"{arm_feedback}. {separation_feedback}."
            
        except Exception as e:
            result["feedback"] = f"Error analyzing arm/shoulder rotation: {str(e)}"
            
        return result
