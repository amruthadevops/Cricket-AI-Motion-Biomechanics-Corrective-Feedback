# src/analyzers/follow_through_analyzer.py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class FollowThroughAnalyzer:
    def __init__(self):
        # Initialize thresholds and parameters
        self.min_confidence = 0.5
        self.balance_threshold = 0.15  # Threshold for detecting balance loss
        self.wrist_motion_threshold = 0.1  # Threshold for wrist motion detection
        self.torque_transfer_threshold = 0.2  # Threshold for torque transfer efficiency
        
    def analyze(self, frames_data: List[Dict]) -> Dict:
        """
        Analyze the follow-through for both batting and bowling.
        
        Args:
            frames_data: List of dictionaries containing frame data with pose keypoints
            
        Returns:
            Dictionary containing follow-through analysis results
        """
        results = {
            "balance": {
                "detected": False,
                "score": 0.0,
                "feedback": ""
            },
            "wrist_motion": {
                "detected": False,
                "path": [],
                "score": 0.0,
                "feedback": ""
            },
            "torque_transfer": {
                "detected": False,
                "efficiency": 0.0,
                "score": 0.0,
                "feedback": ""
            },
            "scores": {
                "balance_score": 0.0,
                "wrist_motion_score": 0.0,
                "torque_transfer_score": 0.0,
                "overall_score": 0.0
            },
            "corrections": []
        }
        
        if not frames_data:
            results["error"] = "No frame data provided"
            return results
            
        try:
            # Detect action type (batting or bowling)
            action_type = self._detect_action_type(frames_data)
            
            # Find follow-through frames
            follow_through_frames = self._find_follow_through_frames(frames_data, action_type)
            
            if not follow_through_frames:
                results["error"] = "No follow-through frames detected"
                return results
            
            # Analyze balance
            balance_results = self._analyze_balance(follow_through_frames)
            results["balance"] = balance_results
            
            # Analyze wrist motion
            wrist_motion_results = self._analyze_wrist_motion(follow_through_frames, action_type)
            results["wrist_motion"] = wrist_motion_results
            
            # Analyze torque transfer
            torque_transfer_results = self._analyze_torque_transfer(follow_through_frames)
            results["torque_transfer"] = torque_transfer_results
            
            # Calculate overall scores
            self._calculate_overall_scores(results)
            
            # Generate corrections
            results["corrections"] = self._generate_corrections(results, action_type)
            
        except Exception as e:
            results["error"] = str(e)
            
        return results
    
    def _detect_action_type(self, frames_data: List[Dict]) -> str:
        """Detect whether the action is batting or bowling."""
        try:
            # Look for key indicators of batting or bowling
            # For batting, we expect to see more bat movement (wrist positions)
            # For bowling, we expect to see more run-up and arm action
            
            # Calculate average wrist movement
            total_wrist_movement = 0
            valid_frames = 0
            
            for i in range(1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_wrist", "right_wrist"]) or \
                   not all(k in curr_keypoints for k in ["left_wrist", "right_wrist"]):
                    continue
                
                # Calculate wrist movement
                prev_left_wrist = prev_keypoints["left_wrist"]
                prev_right_wrist = prev_keypoints["right_wrist"]
                curr_left_wrist = curr_keypoints["left_wrist"]
                curr_right_wrist = curr_keypoints["right_wrist"]
                
                left_wrist_movement = np.sqrt(
                    (curr_left_wrist["x"] - prev_left_wrist["x"])**2 + 
                    (curr_left_wrist["y"] - prev_left_wrist["y"])**2
                )
                
                right_wrist_movement = np.sqrt(
                    (curr_right_wrist["x"] - prev_right_wrist["x"])**2 + 
                    (curr_right_wrist["y"] - prev_right_wrist["y"])**2
                )
                
                total_wrist_movement += left_wrist_movement + right_wrist_movement
                valid_frames += 1

            if valid_frames == 0:
                return "unknown"
            
            avg_wrist_movement = total_wrist_movement / valid_frames
            
            # Calculate average body movement (hip and shoulder)
            total_body_movement = 0
            valid_frames = 0
            
            for i in range(1, len(frames_data)):
                if not frames_data[i-1].get("pose_detected") or not frames_data[i].get("pose_detected"):
                    continue
                    
                prev_keypoints = frames_data[i-1].get("key_points", {})
                curr_keypoints = frames_data[i].get("key_points", {})
                
                if not all(k in prev_keypoints for k in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]) or \
                   not all(k in curr_keypoints for k in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]):
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
                
                hip_movement = np.sqrt(
                    (curr_hip_center["x"] - prev_hip_center["x"])**2 + 
                    (curr_hip_center["y"] - prev_hip_center["y"])**2
                )
                
                # Calculate shoulder movement
                prev_shoulder_center = {
                    "x": (prev_keypoints["left_shoulder"]["x"] + prev_keypoints["right_shoulder"]["x"]) / 2,
                    "y": (prev_keypoints["left_shoulder"]["y"] + prev_keypoints["right_shoulder"]["y"]) / 2
                }
                
                curr_shoulder_center = {
                    "x": (curr_keypoints["left_shoulder"]["x"] + curr_keypoints["right_shoulder"]["x"]) / 2,
                    "y": (curr_keypoints["left_shoulder"]["y"] + curr_keypoints["right_shoulder"]["y"]) / 2
                }
                
                shoulder_movement = np.sqrt(
                    (curr_shoulder_center["x"] - prev_shoulder_center["x"])**2 + 
                    (curr_shoulder_center["y"] - prev_shoulder_center["y"])**2
                )
                
                total_body_movement += hip_movement + shoulder_movement
                valid_frames += 1
            
            if valid_frames == 0:
                return "unknown"
            
            avg_body_movement = total_body_movement / valid_frames
            
            # Determine action type based on movement patterns
            # Batting typically has more wrist movement relative to body movement
            # Bowling typically has more body movement relative to wrist movement
            
            if avg_wrist_movement > avg_body_movement * 1.5:
                return "batting"
            elif avg_body_movement > avg_wrist_movement * 1.5:
                return "bowling"
            else:
                return "unknown"
                
        except Exception as e:
            print(f"Error detecting action type: {str(e)}")
            return "unknown"
    
    def _find_follow_through_frames(self, frames_data: List[Dict], action_type: str) -> List[Dict]:
        """Find the frames corresponding to the follow-through phase."""
        try:
            if action_type == "batting":
                return self._find_batting_follow_through_frames(frames_data)
            elif action_type == "bowling":
                return self._find_bowling_follow_through_frames(frames_data)
            else:
                # Try both and return the one with more frames
                batting_frames = self._find_batting_follow_through_frames(frames_data)
                bowling_frames = self._find_bowling_follow_through_frames(frames_data)
                
                if len(batting_frames) > len(bowling_frames):
                    return batting_frames
                else:
                    return bowling_frames
                    
        except Exception as e:
            print(f"Error finding follow-through frames: {str(e)}")
            return []
    
    def _find_batting_follow_through_frames(self, frames_data: List[Dict]) -> List[Dict]:
        """Find the frames corresponding to the batting follow-through phase."""
        try:
            # Find the impact frame (frame with maximum bat speed)
            impact_frame_idx = self._find_batting_impact_frame(frames_data)
            
            if impact_frame_idx is None:
                return []
            
            # Get frames after impact
            follow_through_frames = frames_data[impact_frame_idx+1:]
            
            # Limit to a reasonable number of frames
            if len(follow_through_frames) > 20:
                follow_through_frames = follow_through_frames[:20]
            
            return follow_through_frames
            
        except Exception as e:
            print(f"Error finding batting follow-through frames: {str(e)}")
            return []
    
    def _find_bowling_follow_through_frames(self, frames_data: List[Dict]) -> List[Dict]:
        """Find the frames corresponding to the bowling follow-through phase."""
        try:
            # Find the delivery frame (frame with maximum arm speed)
            delivery_frame_idx = self._find_bowling_delivery_frame(frames_data)
            
            if delivery_frame_idx is None:
                return []
            
            # Get frames after delivery
            follow_through_frames = frames_data[delivery_frame_idx+1:]
            
            # Limit to a reasonable number of frames
            if len(follow_through_frames) > 20:
                follow_through_frames = follow_through_frames[:20]
            
            return follow_through_frames
            
        except Exception as e:
            print(f"Error finding bowling follow-through frames: {str(e)}")
            return []
    
    def _find_batting_impact_frame(self, frames_data: List[Dict]) -> Optional[int]:
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
            print(f"Error finding batting impact frame: {str(e)}")
            return None
    
    def _find_bowling_delivery_frame(self, frames_data: List[Dict]) -> Optional[int]:
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
            print(f"Error finding bowling delivery frame: {str(e)}")
            return None
    
    def _analyze_balance(self, follow_through_frames: List[Dict]) -> Dict:
        """Analyze the balance during the follow-through."""
        balance_result = {
            "detected": False,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            if not follow_through_frames:
                balance_result["feedback"] = "No follow-through frames provided"
                return balance_result
            
            balance_result["detected"] = True
            
            # Calculate center of gravity for each frame
            cog_positions = []
            for frame in follow_through_frames:
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
                balance_result["feedback"] = "Insufficient data for balance analysis"
                return balance_result
            
            # Calculate stability of center of gravity
            cog_stability = self._calculate_cog_stability(cog_positions)
            
            # Calculate body alignment
            alignment_score = self._calculate_body_alignment(follow_through_frames)
            
            # Calculate overall balance score
            balance_result["score"] = (cog_stability + alignment_score) / 2
            
            # Generate feedback
            if balance_result["score"] > 0.8:
                balance_result["feedback"] = "Excellent balance during follow-through"
            elif balance_result["score"] > 0.6:
                balance_result["feedback"] = "Good balance during follow-through"
            elif balance_result["score"] > 0.4:
                balance_result["feedback"] = "Average balance, could improve stability"
            else:
                balance_result["feedback"] = "Poor balance, focus on stability during follow-through"
                
        except Exception as e:
            balance_result["feedback"] = f"Error analyzing balance: {str(e)}"
            
        return balance_result
    
    def _analyze_wrist_motion(self, follow_through_frames: List[Dict], action_type: str) -> Dict:
        """Analyze the wrist motion during the follow-through."""
        wrist_motion_result = {
            "detected": False,
            "path": [],
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            if not follow_through_frames:
                wrist_motion_result["feedback"] = "No follow-through frames provided"
                return wrist_motion_result
            
            wrist_motion_result["detected"] = True
            
            # Extract wrist positions
            wrist_positions = []
            for frame in follow_through_frames:
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                
                if action_type == "batting":
                    if not all(k in keypoints for k in ["left_wrist", "right_wrist"]):
                        continue
                    
                    # Use both wrists for batting
                    left_wrist = keypoints["left_wrist"]
                    right_wrist = keypoints["right_wrist"]
                    wrist_center = {
                        "x": (left_wrist["x"] + right_wrist["x"]) / 2,
                        "y": (left_wrist["y"] + right_wrist["y"]) / 2
                    }
                    wrist_positions.append(wrist_center)
                    
                elif action_type == "bowling":
                    if "right_wrist" not in keypoints:
                        continue
                    
                    # Use bowling arm wrist for bowling
                    wrist_positions.append(keypoints["right_wrist"])
            
            if len(wrist_positions) < 2:
                wrist_motion_result["feedback"] = "Insufficient wrist position data"
                return wrist_motion_result
            
            # Store wrist path
            wrist_motion_result["path"] = wrist_positions
            
            # Calculate wrist motion smoothness
            smoothness = self._calculate_wrist_smoothness(wrist_positions)
            
            # Calculate wrist extension
            extension = self._calculate_wrist_extension(wrist_positions)
            
            # Calculate overall wrist motion score
            wrist_motion_result["score"] = (smoothness + extension) / 2
            
            # Generate feedback
            if wrist_motion_result["score"] > 0.8:
                wrist_motion_result["feedback"] = "Excellent wrist motion during follow-through"
            elif wrist_motion_result["score"] > 0.6:
                wrist_motion_result["feedback"] = "Good wrist motion during follow-through"
            elif wrist_motion_result["score"] > 0.4:
                wrist_motion_result["feedback"] = "Average wrist motion, could improve fluidity"
            else:
                wrist_motion_result["feedback"] = "Poor wrist motion, focus on smooth follow-through"
                
        except Exception as e:
            wrist_motion_result["feedback"] = f"Error analyzing wrist motion: {str(e)}"
            
        return wrist_motion_result
    
    def _analyze_torque_transfer(self, follow_through_frames: List[Dict]) -> Dict:
        """Analyze the torque transfer efficiency during the follow-through."""
        torque_transfer_result = {
            "detected": False,
            "efficiency": 0.0,
            "score": 0.0,
            "feedback": ""
        }
        
        try:
            if not follow_through_frames:
                torque_transfer_result["feedback"] = "No follow-through frames provided"
                return torque_transfer_result
            
            torque_transfer_result["detected"] = True
            
            # Calculate shoulder-hip torque for each frame
            torque_values = []
            for frame in follow_through_frames:
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                    continue
                
                # Calculate shoulder-hip torque
                torque = self._calculate_shoulder_hip_torque(
                    keypoints["left_shoulder"], 
                    keypoints["right_shoulder"],
                    keypoints["left_hip"], 
                    keypoints["right_hip"]
                )
                torque_values.append(torque)
            
            if len(torque_values) < 2:
                torque_transfer_result["feedback"] = "Insufficient data for torque analysis"
                return torque_transfer_result
            
            # Calculate torque transfer efficiency
            torque_transfer_result["efficiency"] = self._calculate_torque_efficiency(torque_values)
            
            # Calculate score based on efficiency
            torque_transfer_result["score"] = torque_transfer_result["efficiency"]
            
            # Generate feedback
            if torque_transfer_result["score"] > 0.8:
                torque_transfer_result["feedback"] = "Excellent torque transfer during follow-through"
            elif torque_transfer_result["score"] > 0.6:
                torque_transfer_result["feedback"] = "Good torque transfer during follow-through"
            elif torque_transfer_result["score"] > 0.4:
                torque_transfer_result["feedback"] = "Average torque transfer, could improve efficiency"
            else:
                torque_transfer_result["feedback"] = "Poor torque transfer, focus on hip-shoulder separation"
                
        except Exception as e:
            torque_transfer_result["feedback"] = f"Error analyzing torque transfer: {str(e)}"
            
        return torque_transfer_result
    
    def _calculate_overall_scores(self, results: Dict) -> None:
        """Calculate overall scores from individual analysis components."""
        try:
            # Get individual scores
            balance_score = results["balance"]["score"] if results["balance"]["detected"] else 0.5
            wrist_motion_score = results["wrist_motion"]["score"] if results["wrist_motion"]["detected"] else 0.5
            torque_transfer_score = results["torque_transfer"]["score"] if results["torque_transfer"]["detected"] else 0.5
            
            # Calculate weighted average
            overall_score = (
                balance_score * 0.4 +
                wrist_motion_score * 0.3 +
                torque_transfer_score * 0.3
            )
            
            # Update scores dictionary
            results["scores"]["balance_score"] = balance_score
            results["scores"]["wrist_motion_score"] = wrist_motion_score
            results["scores"]["torque_transfer_score"] = torque_transfer_score
            results["scores"]["overall_score"] = overall_score
            
        except Exception as e:
            print(f"Error calculating overall scores: {str(e)}")
    
    def _generate_corrections(self, results: Dict, action_type: str) -> List[str]:
        """Generate corrective feedback based on analysis results."""
        corrections = []
        
        try:
            # Balance corrections
            if results["balance"]["detected"]:
                if results["balance"]["score"] < 0.7:
                    corrections.append("Improve balance during follow-through - maintain stable posture")
            else:
                corrections.append("Focus on balance during follow-through")
            
            # Wrist motion corrections
            if results["wrist_motion"]["detected"]:
                if results["wrist_motion"]["score"] < 0.7:
                    if action_type == "batting":
                        corrections.append("Ensure smooth bat follow-through with relaxed wrists")
                    else:
                        corrections.append("Improve wrist motion for better ball control")
            else:
                corrections.append("Focus on complete wrist follow-through")
            
            # Torque transfer corrections
            if results["torque_transfer"]["detected"]:
                if results["torque_transfer"]["score"] < 0.7:
                    corrections.append("Improve hip-shoulder separation for better power transfer")
            else:
                corrections.append("Focus on torque transfer during follow-through")
            
            # If no specific corrections, provide general encouragement
            if not corrections:
                corrections.append("Good follow-through technique overall - continue practicing!")
                
        except Exception as e:
            corrections.append(f"Error generating corrections: {str(e)}")
            
        return corrections
    
    def _calculate_cog_stability(self, cog_positions: List[Dict]) -> float:
        """Calculate the stability of the center of gravity."""
        try:
            if len(cog_positions) < 2:
                return 0.0
            
            # Calculate variance in COG positions
            x_values = [pos["x"] for pos in cog_positions]
            y_values = [pos["y"] for pos in cog_positions]
            
            x_var = np.var(x_values)
            y_var = np.var(y_values)
            
            # Calculate stability (inverse of variance)
            x_stability = 1.0 / (1.0 + x_var)
            y_stability = 1.0 / (1.0 + y_var)
            
            # Overall stability is the average of x and y stability
            stability = (x_stability + y_stability) / 2
            
            return stability
            
        except Exception as e:
            print(f"Error calculating COG stability: {str(e)}")
            return 0.0
    
    def _calculate_body_alignment(self, frames_data: List[Dict]) -> float:
        """Calculate the body alignment during the follow-through."""
        try:
            if not frames_data:
                return 0.0
            
            alignment_scores = []
            
            for frame in frames_data:
                if not frame.get("pose_detected"):
                    continue
                    
                keypoints = frame.get("key_points", {})
                if not all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                    continue
                
                # Calculate shoulder vector
                shoulder_vector = np.array([
                    keypoints["right_shoulder"]["x"] - keypoints["left_shoulder"]["x"],
                    keypoints["right_shoulder"]["y"] - keypoints["left_shoulder"]["y"]
                ])
                
                # Calculate hip vector
                hip_vector = np.array([
                    keypoints["right_hip"]["x"] - keypoints["left_hip"]["x"],
                    keypoints["right_hip"]["y"] - keypoints["left_hip"]["y"]
                ])
                
                # Normalize vectors
                shoulder_norm = np.linalg.norm(shoulder_vector)
                hip_norm = np.linalg.norm(hip_vector)
                
                if shoulder_norm == 0 or hip_norm == 0:
                    continue
                
                shoulder_vector = shoulder_vector / shoulder_norm
                hip_vector = hip_vector / hip_norm
                
                # Calculate alignment (dot product)
                alignment = np.dot(shoulder_vector, hip_vector)
                
                # Ensure alignment is between 0 and 1
                alignment = max(0, min(1, alignment))
                
                alignment_scores.append(alignment)
            
            if not alignment_scores:
                return 0.0
            
            # Return the average alignment score
            return np.mean(alignment_scores)
            
        except Exception as e:
            print(f"Error calculating body alignment: {str(e)}")
            return 0.0
    
    def _calculate_wrist_smoothness(self, wrist_positions: List[Dict]) -> float:
        """Calculate the smoothness of the wrist motion."""
        try:
            if len(wrist_positions) < 3:
                return 0.0
            
            # Calculate velocities
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
            if not accelerations:
                return 0.0
                
            accel_variance = np.var(accelerations)
            smoothness = 1.0 / (1.0 + accel_variance)
            
            return smoothness
            
        except Exception as e:
            print(f"Error calculating wrist smoothness: {str(e)}")
            return 0.0
    
    def _calculate_wrist_extension(self, wrist_positions: List[Dict]) -> float:
        """Calculate the extension of the wrist motion."""
        try:
            if len(wrist_positions) < 2:
                return 0.0
            
            # Calculate total distance traveled by wrist
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
            if direct_distance == 0:
                return 0.0
                
            extension_ratio = total_distance / direct_distance
            
            # Normalize to 0-1 range
            extension = min(1.0, extension_ratio / 3.0)
            
            return extension
            
        except Exception as e:
            print(f"Error calculating wrist extension: {str(e)}")
            return 0.0
    
    def _calculate_shoulder_hip_torque(self, left_shoulder: Dict, right_shoulder: Dict, 
                                     left_hip: Dict, right_hip: Dict) -> float:
        """Calculate the torque between shoulders and hips."""
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
            print(f"Error calculating shoulder-hip torque: {str(e)}")
            return 0.0
    
    def _calculate_torque_efficiency(self, torque_values: List[float]) -> float:
        """Calculate the efficiency of torque transfer."""
        try:
            if len(torque_values) < 2:
                return 0.0
            
            # Calculate the rate of change of torque
            torque_changes = []
            for i in range(1, len(torque_values)):
                change = abs(torque_values[i] - torque_values[i-1])
                torque_changes.append(change)
            
            # Calculate efficiency (inverse of variance in torque changes)
            if not torque_changes:
                return 0.0
                
            change_variance = np.var(torque_changes)
            efficiency = 1.0 / (1.0 + change_variance)
            
            return efficiency
            
        except Exception as e:
            print(f"Error calculating torque efficiency: {str(e)}")
            return 0.0

