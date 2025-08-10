# visual_feedback_overlay.py
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class OverlayStyle(Enum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"
    SUCCESS = "success"
    INFO = "info"

@dataclass
class VisualElement:
    element_type: str  # "text", "arrow", "circle", "rectangle", "line"
    position: Tuple[int, int]
    style: OverlayStyle
    content: str
    size: int = 1
    duration: int = 30  # frames to display
    animation_type: str = "static"  # "static", "pulse", "fade"

class VisualFeedbackOverlay:
    def __init__(self):
        # Color schemes for different severity levels
        self.color_schemes = {
            OverlayStyle.CRITICAL: {
                'primary': (0, 0, 255),      # Red
                'secondary': (0, 0, 200),     # Dark red
                'background': (0, 0, 255, 80)  # Semi-transparent red
            },
            OverlayStyle.MODERATE: {
                'primary': (0, 165, 255),     # Orange
                'secondary': (0, 140, 255),   # Dark orange
                'background': (0, 165, 255, 80)
            },
            OverlayStyle.MINOR: {
                'primary': (0, 255, 255),     # Yellow
                'secondary': (0, 200, 255),   # Dark yellow
                'background': (0, 255, 255, 80)
            },
            OverlayStyle.SUCCESS: {
                'primary': (0, 255, 0),       # Green
                'secondary': (0, 200, 0),     # Dark green
                'background': (0, 255, 0, 80)
            },
            OverlayStyle.INFO: {
                'primary': (255, 255, 255),   # White
                'secondary': (200, 200, 200), # Light gray
                'background': (255, 255, 255, 80)
            }
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scales = {
            'small': 0.5,
            'medium': 0.7,
            'large': 1.0,
            'extra_large': 1.5
        }
        
        # Animation state tracking
        self.active_elements = []
        self.frame_counter = 0

    def add_fielding_feedback(self, frame: np.ndarray, corrections: List[Dict], 
                            dive_analysis: Dict, throw_analysis: Dict) -> np.ndarray:
        """Add fielding-specific visual feedback to frame"""
        overlay_frame = frame.copy()
        
        # Process fielding corrections
        for correction in corrections:
            if correction.get('correction_type') == 'dive_orientation':
                overlay_frame = self._add_dive_orientation_feedback(
                    overlay_frame, correction, dive_analysis
                )
            elif correction.get('correction_type') == 'dive_timing':
                overlay_frame = self._add_dive_timing_feedback(
                    overlay_frame, correction
                )
            elif correction.get('correction_type') == 'shoulder_alignment':
                overlay_frame = self._add_shoulder_alignment_feedback(
                    overlay_frame, correction, throw_analysis
                )
            elif correction.get('correction_type') == 'arm_lag':
                overlay_frame = self._add_arm_lag_feedback(
                    overlay_frame, correction
                )
        
        # Add overall performance indicators
        overlay_frame = self._add_performance_indicators(
            overlay_frame, dive_analysis, throw_analysis
        )
        
        return overlay_frame

    def add_follow_through_feedback(self, frame: np.ndarray, corrections: List[Dict],
                                  balance_analysis: Dict, cog_analysis: Dict,
                                  wrist_analysis: Dict, torque_analysis: Dict,
                                  action_type: str) -> np.ndarray:
        """Add follow-through specific visual feedback to frame"""
        overlay_frame = frame.copy()
        
        # Process follow-through corrections
        for correction in corrections:
            correction_type = correction.get('correction_type', '')
            
            if correction_type == 'balance_loss':
                overlay_frame = self._add_balance_feedback(
                    overlay_frame, correction, balance_analysis
                )
            elif correction_type == 'cog_instability':
                overlay_frame = self._add_cog_feedback(
                    overlay_frame, correction, cog_analysis
                )
            elif correction_type == 'wrist_stiffness':
                overlay_frame = self._add_wrist_feedback(
                    overlay_frame, correction, wrist_analysis
                )
            elif correction_type == 'incomplete_follow_through':
                overlay_frame = self._add_torque_feedback(
                    overlay_frame, correction, torque_analysis
                )
        
        # Add biomechanical analysis overlay
        overlay_frame = self._add_biomechanical_overlay(
            overlay_frame, balance_analysis, cog_analysis, 
            wrist_analysis, torque_analysis, action_type
        )
        
        return overlay_frame

    def _add_dive_orientation_feedback(self, frame: np.ndarray, correction: Dict, 
                                     dive_analysis: Dict) -> np.ndarray:
        """Add dive orientation visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw attention circle around the dive area
        cv2.circle(frame, pos, 60, self.color_schemes[OverlayStyle.MODERATE]['primary'], 3)
        
        # Add directional arrow showing proper hand extension
        arrow_end = (pos[0] + 80, pos[1] - 30)
        cv2.arrowedLine(frame, pos, arrow_end, 
                       self.color_schemes[OverlayStyle.MODERATE]['primary'], 3)
        
        # Add text feedback
        text = "Extend hands forward"
        text_pos = (pos[0] - 50, pos[1] + 80)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.MODERATE)
        
        # Add orientation score
        score_text = f"Orientation: {dive_analysis.get('orientation_score', 0):.2f}"
        score_pos = (pos[0] - 50, pos[1] + 100)
        self._draw_text_with_background(frame, score_text, score_pos, OverlayStyle.INFO, 'small')
        
        return frame

    def _add_dive_timing_feedback(self, frame: np.ndarray, correction: Dict) -> np.ndarray:
        """Add dive timing visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw pulsing circle to indicate timing issue
        pulse_radius = 40 + int(10 * np.sin(self.frame_counter * 0.3))
        cv2.circle(frame, pos, pulse_radius, self.color_schemes[OverlayStyle.CRITICAL]['primary'], 3)
        
        # Add timing indicator with clock icon simulation
        self._draw_timing_indicator(frame, (pos[0] - 80, pos[1] - 20))
        
        # Add text feedback
        text = "REACT EARLIER!"
        text_pos = (pos[0] - 60, pos[1] + 60)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.CRITICAL, 'medium')
        
        return frame

    def _add_shoulder_alignment_feedback(self, frame: np.ndarray, correction: Dict, 
                                       throw_analysis: Dict) -> np.ndarray:
        """Add shoulder alignment visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw alignment guides
        shoulder_left = (pos[0] - 50, pos[1])
        shoulder_right = (pos[0] + 50, pos[1])
        
        # Current alignment (poor)
        cv2.line(frame, shoulder_left, shoulder_right, 
                self.color_schemes[OverlayStyle.MODERATE]['primary'], 3)
        
        # Ideal alignment (slightly above, in green)
        ideal_left = (shoulder_left[0], shoulder_left[1] - 15)
        ideal_right = (shoulder_right[0], shoulder_right[1] - 15)
        cv2.line(frame, ideal_left, ideal_right, 
                self.color_schemes[OverlayStyle.SUCCESS]['primary'], 2)
        
        # Add rotation arrow
        center = pos
        self._draw_rotation_arrow(frame, center, 45, OverlayStyle.MODERATE)
        
        # Add feedback text
        text = "Align shoulders"
        text_pos = (pos[0] - 50, pos[1] + 50)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.MODERATE)
        
        # Add rotation score
        rotation_score = throw_analysis.get('shoulder_rotation', 0)
        score_text = f"Rotation: {rotation_score:.2f}"
        score_pos = (pos[0] - 50, pos[1] + 70)
        self._draw_text_with_background(frame, score_text, score_pos, OverlayStyle.INFO, 'small')
        
        return frame

    def _add_arm_lag_feedback(self, frame: np.ndarray, correction: Dict) -> np.ndarray:
        """Add arm lag visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw arm motion trail to show lag
        trail_points = [
            (pos[0] - 40, pos[1] + 20),
            (pos[0] - 20, pos[1] + 10),
            pos,
            (pos[0] + 20, pos[1] - 10),
            (pos[0] + 40, pos[1] - 20)
        ]
        
        # Draw motion trail with decreasing opacity
        for i, point in enumerate(trail_points[:-1]):
            alpha = 0.3 + (i * 0.15)
            color = tuple(int(c * alpha) for c in self.color_schemes[OverlayStyle.MODERATE]['primary'])
            cv2.circle(frame, point, 8 - i, color, -1)
        
        # Draw arrow showing desired follow-through direction
        arrow_start = pos
        arrow_end = (pos[0] + 80, pos[1] - 40)
        cv2.arrowedLine(frame, arrow_start, arrow_end,
                       self.color_schemes[OverlayStyle.SUCCESS]['primary'], 3)
        
        # Add text feedback
        text = "Complete follow-through"
        text_pos = (pos[0] - 70, pos[1] + 50)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.MODERATE)
        
        return frame

    def _add_balance_feedback(self, frame: np.ndarray, correction: Dict, 
                            balance_analysis: Dict) -> np.ndarray:
        """Add balance loss visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw balance indicator - tilted rectangle showing instability
        rect_center = pos
        rect_size = (60, 20)
        
        # Calculate tilt angle based on balance score
        balance_score = balance_analysis.get('average_balance_score', 0.5)
        tilt_angle = (1 - balance_score) * 30  # Max 30 degree tilt
        
        # Draw tilted rectangle
        self._draw_tilted_rectangle(frame, rect_center, rect_size, tilt_angle, OverlayStyle.CRITICAL)
        
        # Draw ideal balance position (horizontal rectangle)
        ideal_rect_center = (pos[0], pos[1] - 40)
        self._draw_tilted_rectangle(frame, ideal_rect_center, rect_size, 0, OverlayStyle.SUCCESS)
        
        # Add stability arrows
        if balance_score < 0.5:
            # Show where to shift weight
            arrow_start = (pos[0], pos[1] + 30)
            arrow_end = (pos[0] - 30, pos[1] + 30)  # Shift left for stability
            cv2.arrowedLine(frame, arrow_start, arrow_end,
                           self.color_schemes[OverlayStyle.CRITICAL]['primary'], 3)
        
        # Add text feedback
        text = correction.get('message', 'Balance issue')
        text_pos = (pos[0] - 50, pos[1] + 60)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.CRITICAL)
        
        # Add balance score
        score_text = f"Balance: {balance_score:.2f}"
        score_pos = (pos[0] - 50, pos[1] + 80)
        self._draw_text_with_background(frame, score_text, score_pos, OverlayStyle.INFO, 'small')
        
        return frame

    def _add_cog_feedback(self, frame: np.ndarray, correction: Dict, 
                        cog_analysis: Dict) -> np.ndarray:
        """Add center of gravity visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw center of gravity indicator
        cog_displacement = cog_analysis.get('cog_displacement', 0)
        stability_score = cog_analysis.get('stability_score', 0)
        
        # Draw current CoG position (red circle)
        cv2.circle(frame, pos, 15, self.color_schemes[OverlayStyle.MODERATE]['primary'], 3)
        cv2.circle(frame, pos, 8, self.color_schemes[OverlayStyle.MODERATE]['primary'], -1)
        
        # Draw ideal CoG position (green circle)
        ideal_pos = (pos[0], pos[1] + 20)
        cv2.circle(frame, ideal_pos, 12, self.color_schemes[OverlayStyle.SUCCESS]['primary'], 2)
        
        # Draw displacement vector
        if cog_displacement > 10:
            cv2.arrowedLine(frame, ideal_pos, pos,
                           self.color_schemes[OverlayStyle.MODERATE]['secondary'], 2)
        
        # Add stability zone indicator
        cv2.circle(frame, ideal_pos, 25, self.color_schemes[OverlayStyle.SUCCESS]['secondary'], 1)
        
        # Add text feedback
        text = "Center of gravity unstable"
        text_pos = (pos[0] - 80, pos[1] + 50)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.MODERATE)
        
        # Add biomechanical insight
        insight = correction.get('biomechanical_insight', '')
        if insight:
            insight_pos = (pos[0] - 90, pos[1] + 70)
            self._draw_text_with_background(frame, insight, insight_pos, OverlayStyle.INFO, 'small')
        
        return frame

    def _add_wrist_feedback(self, frame: np.ndarray, correction: Dict, 
                          wrist_analysis: Dict) -> np.ndarray:
        """Add wrist motion visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw wrist stiffness indicator
        stiffness_percentage = wrist_analysis.get('stiffness_percentage', 0)
        
        # Draw current wrist position with stiffness indicator
        if stiffness_percentage > 0.4:
            # Red square for stiff wrist
            cv2.rectangle(frame, (pos[0] - 10, pos[1] - 10), 
                         (pos[0] + 10, pos[1] + 10),
                         self.color_schemes[OverlayStyle.MODERATE]['primary'], 3)
            
            # Draw desired fluid motion path
            motion_path = [
                pos,
                (pos[0] + 20, pos[1] - 10),
                (pos[0] + 40, pos[1] - 15),
                (pos[0] + 60, pos[1] - 10)
            ]
            
            for i in range(len(motion_path) - 1):
                cv2.line(frame, motion_path[i], motion_path[i + 1],
                        self.color_schemes[OverlayStyle.SUCCESS]['primary'], 2)
                # Add small circles along the path
                cv2.circle(frame, motion_path[i + 1], 3,
                          self.color_schemes[OverlayStyle.SUCCESS]['primary'], -1)
        
        # Add text feedback
        text = "Allow wrist to flow naturally"
        text_pos = (pos[0] - 80, pos[1] + 40)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.MODERATE)
        
        # Add motion quality score
        motion_quality = wrist_analysis.get('motion_quality', 0)
        score_text = f"Motion Quality: {motion_quality:.2f}"
        score_pos = (pos[0] - 80, pos[1] + 60)
        self._draw_text_with_background(frame, score_text, score_pos, OverlayStyle.INFO, 'small')
        
        return frame

    def _add_torque_feedback(self, frame: np.ndarray, correction: Dict, 
                           torque_analysis: Dict) -> np.ndarray:
        """Add torque transfer visual feedback"""
        pos = correction.get('position', (50, 50))
        
        # Draw torque transfer visualization
        efficiency = torque_analysis.get('average_efficiency', 0)
        
        # Draw hip and shoulder rotation indicators
        hip_center = (pos[0], pos[1] + 40)
        shoulder_center = (pos[0], pos[1] - 20)
        
        # Draw hip rotation (base of kinetic chain)
        self._draw_rotation_indicator(frame, hip_center, 30, 0.6, OverlayStyle.INFO, "HIPS")
        
        # Draw shoulder rotation (end of kinetic chain)
        rotation_offset = efficiency * 45  # More efficient = more rotation
        self._draw_rotation_indicator(frame, shoulder_center, 25, efficiency, 
                                    OverlayStyle.SUCCESS if efficiency > 0.6 else OverlayStyle.MODERATE, 
                                    "SHOULDERS")
        
        # Draw connection line showing energy transfer
        line_color = (self.color_schemes[OverlayStyle.SUCCESS]['primary'] if efficiency > 0.6 
                     else self.color_schemes[OverlayStyle.MODERATE]['primary'])
        cv2.line(frame, hip_center, shoulder_center, line_color, 3)
        
        # Add efficiency arrows
        if efficiency < 0.6:
            # Show direction for better transfer
            arrow_start = (hip_center[0] + 40, hip_center[1])
            arrow_end = (shoulder_center[0] + 40, shoulder_center[1])
            cv2.arrowedLine(frame, arrow_start, arrow_end,
                           self.color_schemes[OverlayStyle.MODERATE]['primary'], 2)
        
        # Add text feedback
        text = "Improve hip-to-shoulder sequence"
        text_pos = (pos[0] - 90, pos[1] + 80)
        self._draw_text_with_background(frame, text, text_pos, OverlayStyle.MODERATE)
        
        # Add efficiency score
        score_text = f"Transfer Efficiency: {efficiency:.2f}"
        score_pos = (pos[0] - 90, pos[1] + 100)
        self._draw_text_with_background(frame, score_text, score_pos, OverlayStyle.INFO, 'small')
        
        return frame

    def _add_performance_indicators(self, frame: np.ndarray, dive_analysis: Dict, 
                                  throw_analysis: Dict) -> np.ndarray:
        """Add overall performance indicators"""
        # Performance dashboard in top-right corner
        dashboard_x = frame.shape[1] - 200
        dashboard_y = 30
        
        # Background for dashboard
        cv2.rectangle(frame, (dashboard_x - 10, dashboard_y - 10),
                     (dashboard_x + 180, dashboard_y + 120),
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (dashboard_x - 10, dashboard_y - 10),
                     (dashboard_x + 180, dashboard_y + 120),
                     (255, 255, 255), 2)  # White border
        
        # Title
        cv2.putText(frame, "FIELDING ANALYSIS", (dashboard_x, dashboard_y),
                   self.font, 0.5, (255, 255, 255), 1)
        
        # Dive quality
        dive_quality = dive_analysis.get('quality', 'unknown')
        quality_color = self._get_quality_color(dive_quality)
        cv2.putText(frame, f"Dive: {dive_quality.upper()}", 
                   (dashboard_x, dashboard_y + 25),
                   self.font, 0.4, quality_color, 1)
        
        # Throw technique scores
        shoulder_rotation = throw_analysis.get('shoulder_rotation', 0)
        wrist_extension = throw_analysis.get('wrist_extension', 0)
        
        cv2.putText(frame, f"Shoulder: {shoulder_rotation:.2f}", 
                   (dashboard_x, dashboard_y + 45),
                   self.font, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Wrist: {wrist_extension:.2f}", 
                   (dashboard_x, dashboard_y + 65),
                   self.font, 0.4, (255, 255, 255), 1)
        
        # Overall performance bar
        overall_score = (shoulder_rotation + wrist_extension + 
                        (0.8 if dive_quality == 'good' else 0.5 if dive_quality == 'poor' else 0.3)) / 3
        
        bar_width = int(150 * overall_score)
        bar_color = self._get_score_color(overall_score)
        
        cv2.rectangle(frame, (dashboard_x, dashboard_y + 85),
                     (dashboard_x + bar_width, dashboard_y + 95),
                     bar_color, -1)
        cv2.rectangle(frame, (dashboard_x, dashboard_y + 85),
                     (dashboard_x + 150, dashboard_y + 95),
                     (255, 255, 255), 1)
        
        cv2.putText(frame, f"Overall: {overall_score:.2f}", 
                   (dashboard_x, dashboard_y + 110),
                   self.font, 0.4, (255, 255, 255), 1)
        
        return frame

    def _add_biomechanical_overlay(self, frame: np.ndarray, balance_analysis: Dict,
                                 cog_analysis: Dict, wrist_analysis: Dict,
                                 torque_analysis: Dict, action_type: str) -> np.ndarray:
        """Add comprehensive biomechanical analysis overlay"""
        # Biomechanical dashboard in bottom-left corner
        dashboard_x = 30
        dashboard_y = frame.shape[0] - 150
        
        # Background
        cv2.rectangle(frame, (dashboard_x - 10, dashboard_y - 30),
                     (dashboard_x + 250, dashboard_y + 120),
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (dashboard_x - 10, dashboard_y - 30),
                     (dashboard_x + 250, dashboard_y + 120),
                     (255, 255, 255), 2)  # White border
        
        # Title
        title = f"{action_type.upper()} BIOMECHANICS"
        cv2.putText(frame, title, (dashboard_x, dashboard_y - 10),
                   self.font, 0.5, (255, 255, 255), 1)
        
        # Balance analysis
        balance_score = balance_analysis.get('average_balance_score', 0)
        balance_color = self._get_score_color(balance_score)
        cv2.putText(frame, f"Balance: {balance_score:.2f}", 
                   (dashboard_x, dashboard_y + 15),
                   self.font, 0.4, balance_color, 1)
        
        # CoG stability
        cog_stability = cog_analysis.get('stability_score', 0)
        cog_color = self._get_score_color(cog_stability)
        cv2.putText(frame, f"CoG Stability: {cog_stability:.2f}", 
                   (dashboard_x, dashboard_y + 35),
                   self.font, 0.4, cog_color, 1)
        
        # Wrist motion
        wrist_quality = wrist_analysis.get('motion_quality', 0)
        wrist_color = self._get_score_color(wrist_quality)
        cv2.putText(frame, f"Wrist Motion: {wrist_quality:.2f}", 
                   (dashboard_x, dashboard_y + 55),
                   self.font, 0.4, wrist_color, 1)
        
        # Torque transfer
        torque_efficiency = torque_analysis.get('average_efficiency', 0)
        torque_color = self._get_score_color(torque_efficiency)
        cv2.putText(frame, f"Torque Transfer: {torque_efficiency:.2f}", 
                   (dashboard_x, dashboard_y + 75),
                   self.font, 0.4, torque_color, 1)
        
        # Overall biomechanical score
        overall_bio_score = (balance_score + cog_stability + wrist_quality + torque_efficiency) / 4
        overall_color = self._get_score_color(overall_bio_score)
        cv2.putText(frame, f"Biomech Score: {overall_bio_score:.2f}", 
                   (dashboard_x, dashboard_y + 100),
                   self.font, 0.5, overall_color, 2)
        
        return frame

    # Helper methods for drawing complex shapes and indicators
    def _draw_text_with_background(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                                 style: OverlayStyle, font_size: str = 'medium') -> None:
        """Draw text with background for better readability"""
        font_scale = self.font_scales[font_size]
        thickness = 1 if font_size == 'small' else 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, self.font, font_scale, thickness)
        
        # Draw background rectangle
        bg_color = self.color_schemes[style]['background'][:3]  # Remove alpha for cv2
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_height - 5),
                     (position[0] + text_width + 5, position[1] + baseline + 5),
                     bg_color, -1)
        
        # Draw text
        text_color = self.color_schemes[style]['primary']
        cv2.putText(frame, text, position, self.font, font_scale, text_color, thickness)

    def _draw_timing_indicator(self, frame: np.ndarray, position: Tuple[int, int]) -> None:
        """Draw a clock-like timing indicator"""
        center = position
        radius = 20
        
        # Draw clock circle
        cv2.circle(frame, center, radius, self.color_schemes[OverlayStyle.CRITICAL]['primary'], 2)
        
        # Draw clock hands (showing "late" timing)
        # Hour hand
        hour_end = (center[0], center[1] - radius // 2)
        cv2.line(frame, center, hour_end, self.color_schemes[OverlayStyle.CRITICAL]['primary'], 3)
        
        # Minute hand (pointing to show lateness)
        minute_end = (center[0] + radius // 3 * 2, center[1] - radius // 3)
        cv2.line(frame, center, minute_end, self.color_schemes[OverlayStyle.CRITICAL]['primary'], 2)

    def _draw_rotation_arrow(self, frame: np.ndarray, center: Tuple[int, int], 
                           angle_degrees: float, style: OverlayStyle) -> None:
        """Draw a curved arrow to indicate rotation"""
        radius = 30
        angle_rad = np.radians(angle_degrees)
        
        # Draw curved arrow using multiple line segments
        num_segments = 8
        color = self.color_schemes[style]['primary']
        
        for i in range(num_segments):
            start_angle = i * angle_rad / num_segments
            end_angle = (i + 1) * angle_rad / num_segments
            
            start_point = (int(center[0] + radius * np.cos(start_angle)),
                          int(center[1] + radius * np.sin(start_angle)))
            end_point = (int(center[0] + radius * np.cos(end_angle)),
                        int(center[1] + radius * np.sin(end_angle)))
            
            cv2.line(frame, start_point, end_point, color, 2)
        
        # Add arrowhead
        arrow_tip = (int(center[0] + radius * np.cos(angle_rad)),
                    int(center[1] + radius * np.sin(angle_rad)))
        arrow_base1 = (int(arrow_tip[0] - 10 * np.cos(angle_rad + 0.5)),
                      int(arrow_tip[1] - 10 * np.sin(angle_rad + 0.5)))
        arrow_base2 = (int(arrow_tip[0] - 10 * np.cos(angle_rad - 0.5)),
                      int(arrow_tip[1] - 10 * np.sin(angle_rad - 0.5)))
        
        cv2.line(frame, arrow_tip, arrow_base1, color, 2)
        cv2.line(frame, arrow_tip, arrow_base2, color, 2)

    def _draw_tilted_rectangle(self, frame: np.ndarray, center: Tuple[int, int],
                             size: Tuple[int, int], angle_degrees: float, 
                             style: OverlayStyle) -> None:
        """Draw a tilted rectangle to show balance/alignment"""
        width, height = size
        angle_rad = np.radians(angle_degrees)
        
        # Calculate corner points
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Half dimensions
        hw, hh = width // 2, height // 2
        
        # Corner offsets
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        
        # Rotate and translate corners
        rotated_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a + center[0]
            ry = x * sin_a + y * cos_a + center[1]
            rotated_corners.append((int(rx), int(ry)))
        
        # Draw rectangle
        color = self.color_schemes[style]['primary']
        for i in range(4):
            start = rotated_corners[i]
            end = rotated_corners[(i + 1) % 4]
            cv2.line(frame, start, end, color, 3)

    def _draw_rotation_indicator(self, frame: np.ndarray, center: Tuple[int, int],
                               radius: int, efficiency: float, style: OverlayStyle,
                               label: str) -> None:
        """Draw rotation indicator with efficiency visualization"""
        color = self.color_schemes[style]['primary']
        
        # Draw base circle
        cv2.circle(frame, center, radius, color, 2)
        
        # Draw efficiency arc
        arc_angle = int(360 * efficiency)
        if arc_angle > 0:
            # Draw filled arc to show efficiency
            axes = (radius - 5, radius - 5)
            cv2.ellipse(frame, center, axes, 0, -90, -90 + arc_angle, color, 3)
        
        # Add center dot
        cv2.circle(frame, center, 3, color, -1)
        
        # Add label
        label_pos = (center[0] - len(label) * 4, center[1] + radius + 15)
        cv2.putText(frame, label, label_pos, self.font, 0.3, color, 1)

    def _get_quality_color(self, quality: str) -> Tuple[int, int, int]:
        """Get color based on quality assessment"""
        quality_colors = {
            'excellent': (0, 255, 0),      # Green
            'good': (0, 255, 255),         # Yellow
            'poor': (0, 165, 255),         # Orange
            'no_dive': (128, 128, 128),    # Gray
            'unknown': (255, 255, 255)     # White
        }
        return quality_colors.get(quality.lower(), (255, 255, 255))

    def _get_score_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on numerical score (0-1)"""
        if score >= 0.8:
            return (0, 255, 0)      # Green
        elif score >= 0.6:
            return (0, 255, 255)    # Yellow
        elif score >= 0.4:
            return (0, 165, 255)    # Orange
        else:
            return (0, 0, 255)      # Red

    def update_frame_counter(self) -> None:
        """Update frame counter for animations"""
        self.frame_counter += 1

    def reset_frame_counter(self) -> None:
        """Reset frame counter"""
        self.frame_counter = 0