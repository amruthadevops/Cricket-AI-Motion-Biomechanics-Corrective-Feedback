import numpy as np

class CricketActivityClassifier:
    def classify_activity(self, frames):
        """
        Enhanced cricket activity classification based on biomechanical analysis
        """
        valid = [f for f in frames if f['pose_detected'] and f['key_points']]
        if len(valid) < 10:
            return "unknown"

        # Extract comprehensive movement features
        features = self._extract_features(valid)
        
        # Calculate activity scores using weighted criteria
        batting_score = self._calculate_batting_score(features)
        bowling_score = self._calculate_bowling_score(features)
        fielding_score = self._calculate_fielding_score(features)
        
        # Debug output
        print(f"🏏 Batting score: {batting_score:.3f}")
        print(f"🎳 Bowling score: {bowling_score:.3f}")
        print(f"🥅 Fielding score: {fielding_score:.3f}")
        
        # Determine activity based on highest score with minimum threshold
        max_score = max(batting_score, bowling_score, fielding_score)
        
        if max_score < 0.3:  # Minimum confidence threshold
            return "general"
        elif max_score == batting_score:
            return "batting"
        elif max_score == bowling_score:
            return "bowling"
        else:
            return "fielding"
    
    def _extract_features(self, frames):
        """Extract comprehensive movement features from pose data"""
        features = {
            'wrist_positions': [],
            'hip_positions': [],
            'shoulder_positions': [],
            'ankle_positions': [],
            'elbow_positions': [],
            'knee_positions': []
        }
        
        for f in frames:
            kp = f['key_points']
            if not kp:
                continue
                
            # Extract key body points
            if all(point in kp and kp[point] for point in ['right_wrist', 'left_hip', 'right_hip']):
                features['wrist_positions'].append([kp['right_wrist']['x'], kp['right_wrist']['y']])
                features['hip_positions'].append([
                    (kp['left_hip']['x'] + kp['right_hip']['x']) / 2,
                    (kp['left_hip']['y'] + kp['right_hip']['y']) / 2
                ])
            
            if 'right_shoulder' in kp and kp['right_shoulder']:
                features['shoulder_positions'].append([kp['right_shoulder']['x'], kp['right_shoulder']['y']])
            
            if 'right_ankle' in kp and kp['right_ankle']:
                features['ankle_positions'].append([kp['right_ankle']['x'], kp['right_ankle']['y']])
                
            if 'right_elbow' in kp and kp['right_elbow']:
                features['elbow_positions'].append([kp['right_elbow']['x'], kp['right_elbow']['y']])
                
            if 'right_knee' in kp and kp['right_knee']:
                features['knee_positions'].append([kp['right_knee']['x'], kp['right_knee']['y']])
        
        # Convert to numpy arrays
        for key in features:
            if features[key]:
                features[key] = np.array(features[key])
            else:
                features[key] = np.array([]).reshape(0, 2)
        
        return features
    
    def _calculate_batting_score(self, features):
        """Calculate batting activity score based on characteristic movements"""
        if len(features['wrist_positions']) < 5:
            return 0.0
        
        score = 0.0
        
        # Stance stability (batters have relatively stable hip position)
        if len(features['hip_positions']) > 0:
            hip_stability = 1.0 - np.std(features['hip_positions'][:, 0])
            score += max(0, hip_stability) * 0.3
        
        # Bat-like swing pattern (high wrist acceleration in horizontal plane)
        wrist = features['wrist_positions']
        if len(wrist) > 3:
            wrist_vel = np.diff(wrist, axis=0)
            wrist_accel = np.diff(wrist_vel, axis=0)
            horizontal_accel = np.mean(np.abs(wrist_accel[:, 0]))
            score += min(horizontal_accel * 10, 1.0) * 0.4
        
        # Trigger movement (subtle initial movement before swing)
        if len(features['hip_positions']) > 5:
            early_hip_movement = np.std(features['hip_positions'][:5, 0])
            score += min(early_hip_movement * 5, 1.0) * 0.2
        
        # Characteristic batting stance (moderate hip height)
        if len(features['hip_positions']) > 0:
            avg_hip_height = np.mean(features['hip_positions'][:, 1])
            if 0.4 < avg_hip_height < 0.8:  # Typical batting stance range
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_bowling_score(self, features):
        """Calculate bowling activity score based on characteristic movements"""
        if len(features['hip_positions']) < 5:
            return 0.0
        
        score = 0.0
        
        # Run-up consistency (forward movement pattern)
        hips = features['hip_positions']
        hip_x_diff = np.diff(hips[:, 0])
        forward_movement_ratio = len([x for x in hip_x_diff if x > 0.01]) / len(hip_x_diff)
        if forward_movement_ratio > 0.5:
            score += forward_movement_ratio * 0.4
        
        # High arm swing (characteristic bowling arm motion)
        if len(features['wrist_positions']) > 0:
            wrist_range_y = np.max(features['wrist_positions'][:, 1]) - np.min(features['wrist_positions'][:, 1])
            score += min(wrist_range_y * 2, 1.0) * 0.3
        
        # Front foot landing pattern (sudden deceleration)
        if len(features['ankle_positions']) > 3:
            ankle_vel = np.diff(features['ankle_positions'], axis=0)
            ankle_decel = np.diff(ankle_vel, axis=0)
            max_decel = np.max(np.abs(ankle_decel[:, 0])) if len(ankle_decel) > 0 else 0
            score += min(max_decel * 5, 1.0) * 0.2
        
        # Release point detection (high wrist position)
        if len(features['wrist_positions']) > 0:
            max_wrist_height = 1.0 - np.min(features['wrist_positions'][:, 1])  # Invert Y for height
            score += min(max_wrist_height, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def _calculate_fielding_score(self, features):
        """Calculate fielding activity score based on characteristic movements"""
        if len(features['hip_positions']) < 5:
            return 0.0
        
        score = 0.0
        
        # Lateral movement (fielders move side to side)
        hips = features['hip_positions']
        lateral_movement = np.std(hips[:, 0])
        score += min(lateral_movement * 3, 1.0) * 0.3
        
        # Low stance/ready position
        avg_hip_height = np.mean(hips[:, 1])
        if avg_hip_height > 0.6:  # Lower stance indicates fielding position
            score += 0.2
        
        # Anticipation reaction (initial stillness followed by quick movement)
        if len(hips) > 10:
            early_movement = np.std(hips[:5, :])  # First 25% of frames
            late_movement = np.std(hips[-5:, :])  # Last 25% of frames
            if early_movement < late_movement:  # Still then active
                score += 0.3
        
        # Dive mechanics (sudden large movement in any direction)
        if len(features['wrist_positions']) > 3:
            wrist_vel = np.diff(features['wrist_positions'], axis=0)
            max_velocity = np.max(np.linalg.norm(wrist_vel, axis=1)) if len(wrist_vel) > 0 else 0
            score += min(max_velocity * 2, 1.0) * 0.1
        
        # Throwing motion (if present - high elbow position)
        if len(features['elbow_positions']) > 0 and len(features['shoulder_positions']) > 0:
            # Check for overhead throwing motion
            if len(features['elbow_positions']) == len(features['shoulder_positions']):
                elbow_above_shoulder = np.mean([
                    1 if elbow[1] < shoulder[1] else 0  # Y inverted - lower Y = higher position
                    for elbow, shoulder in zip(features['elbow_positions'], features['shoulder_positions'])
                ])
                score += elbow_above_shoulder * 0.1
        
        return min(score, 1.0)

# Backward compatibility - keeping original simple version as backup
class SimpleCricketActivityClassifier:
    def classify_activity(self, frames):
        valid = [f for f in frames if f['pose_detected']]
        if len(valid) < 10:
            return "unknown"

        m = self._analyze(valid)

        if m['has_run_up'] and m['arm_swing_intensity'] > 0.3:
            return "bowling"
        elif m['lateral_movement'] > 0.2 and m['low_stance']:
            return "fielding"
        elif m['bat_like_swing'] or m['stance_stability'] > 0.7:
            return "batting"
        else:
            return "general"

    def _analyze(self, frames):
        wrist, hips = [], []

        for f in frames:
            kp = f['key_points']
            if not kp: continue
            if all(point in kp and kp[point] for point in ['right_wrist', 'left_hip', 'right_hip']):
                wrist.append([kp['right_wrist']['x'], kp['right_wrist']['y']])
                hips.append([(kp['left_hip']['x'] + kp['right_hip']['x']) / 2,
                             (kp['left_hip']['y'] + kp['right_hip']['y']) / 2])

        if len(wrist) < 5:
            return {
                'has_run_up': False,
                'arm_swing_intensity': 0.0,
                'lateral_movement': 0.0,
                'low_stance': False,
                'bat_like_swing': False,
                'stance_stability': 0.0
            }

        wrist = np.array(wrist)
        hips = np.array(hips)

        dx = np.diff(hips[:, 0])
        run_up = len([x for x in dx if x > 0.01]) > 0.6 * len(dx) if len(dx) > 0 else False

        swing_y = np.max(wrist[:, 1]) - np.min(wrist[:, 1])
        lateral = np.std(hips[:, 0])
        hip_y_avg = np.mean(hips[:, 1])

        vel = np.diff(wrist, axis=0)
        accel = np.sum(np.diff(vel, axis=0)**2) if len(vel) > 1 else 0
        bat_like = accel > 0.05
        stance = 1.0 - lateral if lateral > 0 else 0.5

        return {
            'has_run_up': run_up,
            'arm_swing_intensity': swing_y,
            'lateral_movement': lateral,
            'low_stance': hip_y_avg > 0.6,
            'bat_like_swing': bat_like,
            'stance_stability': stance
        }