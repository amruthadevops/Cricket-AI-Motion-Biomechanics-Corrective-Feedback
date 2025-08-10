# main.py
import sys
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import cv2
import argparse

from src.pose_estimator import CricketPoseEstimator
from src.video_processor import CricketVideoProcessor
from src.activity_classifier import CricketActivityClassifier
from src.analyzers.batting_analyzer import BattingAnalyzer
from src.analyzers.bowling_analyzer import BowlingAnalyzer
from src.analyzers.fielding_analyzer import FieldingAnalyzer
from src.analyzers.follow_through import FollowThroughAnalyzer
from src.utils.visualization import CricketVisualizer
from src.utils.static_plotter import plot_swing_trajectory
from src.utils.correction_visualizer import CricketCorrectionVisualizer
from src.utils.submission_report_generator import SubmissionReportGenerator
from src.utils.metrics import calculate_bat_speed
from src.utils.json_utils import convert_numpy_to_json_serializable
#from src.utils.video_annotator import CricketFinalVideoGenerator  
# from src.utils.progress_tracker import ProgressTracker
from src.visualization.visualization import Visualizer
from src.visualization.video_annotator import VideoAnnotator
from src.visualization.video_overlay_corrector import VideoOverlayCorrector
from src.visualization.correction_visualizer import CorrectionVisualizer


def analyze_cricket_mechanics(frames_data: List[Dict], ball_data_sequence: Optional[List] = None) -> Dict[str, Any]:
    """
    Comprehensive cricket mechanics analysis covering all four key areas.
    
    Args:
        frames_data: List of frame data dictionaries
        ball_data_sequence: Optional list of ball tracking data
        
    Returns:
        Dictionary containing analysis results for all mechanics
    """
    results = {
        "batting_mechanics": {},
        "bowling_mechanics": {},
        "fielding_mechanics": {},
        "follow_through": {}
    }

    if not frames_data:
        print("❌ No frame data provided for analysis")
        return results

    # Batting Mechanics Analysis
    try:
        batting_analyzer = BattingAnalyzer()
        batting_results = batting_analyzer.analyze(frames_data)
        results["batting_mechanics"] = batting_results
    except Exception as e:
        results["batting_mechanics"]["error"] = str(e)

    # Bowling Mechanics Analysis
    try:
        bowling_analyzer = BowlingAnalyzer()
        bowling_results = bowling_analyzer.analyze(frames_data)
        results["bowling_mechanics"] = bowling_results
    except Exception as e:
        results["bowling_mechanics"]["error"] = str(e)

    # Fielding Analysis
    try:
        fielding_analyzer = FieldingAnalyzer()
        fielding_results = fielding_analyzer.analyze(frames_data)
        results["fielding_mechanics"] = fielding_results
    except Exception as e:
        results["fielding_mechanics"]["error"] = str(e)

    # Follow-through Analysis
    try:
        follow_analyzer = FollowThroughAnalyzer()
        follow_results = follow_analyzer.analyze(frames_data)
        results["follow_through"] = follow_results
    except Exception as e:
        results["follow_through"]["error"] = str(e)

    return results


def analyze_cricket_video(video_path: str, video_id: str, create_3d_videos: bool = True) -> Tuple[Optional[Dict], str, Optional[str]]:
    """
    Analyze a cricket video and generate comprehensive results.
    
    Args:
        video_path: Path to the video file
        video_id: Unique identifier for the video
        create_3d_videos: Whether to create 3D visualization videos
        
    Returns:
        Tuple of (results_dict, primary_activity, video_path)
    """
    print(f"\n📽️ Processing {video_path} (ID: {video_id})...")
    print("="*60)

    # Validate video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None, "file_not_found", video_path

    # Initialize components
    try:
        processor = CricketVideoProcessor(skip_frames=2)
        classifier = CricketActivityClassifier()
        visualizer = CricketVisualizer()
        correction_visualizer = CricketCorrectionVisualizer()
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        return None, "initialization_error", video_path
    
    # Initialize 3D visualization components
    video_annotator = None
    video_overlay_corrector = None
    
    if create_3d_videos:
        try:
            video_annotator = VideoAnnotator(output_path="outputs/videos")
            video_overlay_corrector = VideoOverlayCorrector("outputs/videos")
        except Exception as e:
            create_3d_videos = False
    
    # Ensure output directories exist
    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("outputs/images", exist_ok=True)

    frames_data = []
    ball_data_sequence = []
    stats = {}

    # Process video with proper error handling
    try:
        print("🎬 Processing video frames...")
        processed_frames = processor.process_video(video_path, max_frames=200)

        if not processed_frames:
            print(f"❌ No frame data extracted from {video_path}")
            return None, "no_data", video_path

        # Separate pose and ball data with validation
        for frame_info in processed_frames:
            if not isinstance(frame_info, dict):
                continue
                
            # Check for valid pose data
            if (frame_info.get("pose_detected", False) and 
                frame_info.get("landmarks") is not None and 
                hasattr(frame_info["landmarks"], 'size') and 
                frame_info["landmarks"].size > 0):
                frames_data.append(frame_info)
                
            # Extract ball data if available
            ball_data = frame_info.get("ball_data")
            if ball_data is not None:
                ball_data_sequence.append(ball_data)

        # Get processing statistics
        try:
            stats = processor.get_frame_statistics(frames_data)
        except Exception:
            stats = {'valid_poses': len(frames_data), 'total_frames': len(processed_frames)}

    except Exception as e:
        print(f"❌ Error processing video {video_path}: {e}")
        return None, "processing_error", video_path

    if not frames_data:
        print(f"❌ No valid pose data for analysis from {video_path}")
        return None, "no_pose_data", video_path

    # Classify primary activity
    try:
        primary_activity = classifier.classify_activity(frames_data)
        print(f"🏷️ Primary activity detected: {primary_activity}")
    except Exception:
        primary_activity = "unknown"

    # Run comprehensive mechanics analysis
    try:
        comprehensive_results = analyze_cricket_mechanics(frames_data, ball_data_sequence)
        if not comprehensive_results:
            comprehensive_results = {}
    except Exception:
        comprehensive_results = {}

    # Extract data for visualizations
    wrist_positions = []
    landmarks_sequence = []

    try:
        for frame_data in frames_data:
            if not isinstance(frame_data, dict):
                continue
                
            # Extract landmarks for 3D visualization
            if (frame_data.get("pose_detected", False) and 
                frame_data.get("landmarks") is not None):
                landmarks = frame_data["landmarks"]
                if isinstance(landmarks, np.ndarray) and landmarks.size > 0:
                    landmarks_sequence.append(landmarks)

            # Extract wrist positions for swing plotting
            if frame_data.get("pose_detected", False) and frame_data.get("key_points"):
                key_points = frame_data["key_points"]
                if isinstance(key_points, dict) and key_points.get("right_wrist"):
                    rw = key_points["right_wrist"]
                    if isinstance(rw, dict) and 'x' in rw and 'y' in rw:
                        wrist_positions.append([float(rw["x"]), float(rw["y"])])
                    elif isinstance(rw, np.ndarray) and len(rw) >= 2:
                        wrist_positions.append([float(rw[0]), float(rw[1])])

    except Exception:
        pass

    # Generate enhanced visualizations
    try:
        # Swing trajectory plot
        if wrist_positions:
            try:
                plot_swing_trajectory(wrist_positions, f"outputs/swing_plot_{video_id}.png")
            except Exception:
                pass

        # 3D skeleton visualization
        if landmarks_sequence:
            try:
                fig3d = visualizer.create_3d_skeleton(
                    landmarks_sequence[0],
                    title=f"3D Pose Analysis - {primary_activity.title()}"
                )
                if fig3d:
                    fig3d.write_html(f"outputs/skeleton_{video_id}.html")
            except Exception:
                pass

        # 3D Pose Comparison with Ideal
        if landmarks_sequence:
            try:
                comparison_fig = correction_visualizer.create_pose_comparison_3d(
                    landmarks_sequence[0],
                    primary_activity,
                    comprehensive_results
                )
                if comparison_fig:
                    comparison_fig.write_html(f"outputs/pose_comparison_{video_id}.html")
            except Exception:
                pass

        # Comprehensive Correction Dashboard
        try:
            dashboard_fig = correction_visualizer.create_correction_dashboard(
                comprehensive_results,
                frames_data,
                primary_activity
            )
            if dashboard_fig:
                dashboard_fig.write_html(f"outputs/correction_dashboard_{video_id}.html")
        except Exception:
            pass

        # Save detailed correction analysis
        try:
            correction_visualizer.save_correction_analysis(
                comprehensive_results,
                primary_activity,
                video_path,
                output_dir="outputs"
            )
        except Exception:
            pass
            
        # Create 3D visualization videos if requested
        videos_created = {}
        
        if create_3d_videos and landmarks_sequence and video_annotator and video_overlay_corrector:
            try:
                pose_estimator = CricketPoseEstimator()
                
                # Create annotated video with 2D pose and feedback
                try:
                    annotated_video_path = f"outputs/videos/{video_id}_annotated.mp4"
                    video_annotator.create_annotated_video(
                        video_path,
                        annotated_video_path,
                        pose_estimator,
                        correction_visualizer,
                        sample_rate=2
                    )
                    videos_created["annotated"] = annotated_video_path
                except Exception:
                    pass
                
                # Create 3D overlay video
                try:
                    overlay_video_path = f"outputs/videos/{video_id}_3d_overlay.mp4"
                    ideal_keypoints = get_ideal_keypoints(primary_activity)
                    if ideal_keypoints is not None:
                        video_overlay_corrector.create_overlay_video(
                            video_path,
                            overlay_video_path,
                            pose_estimator,
                            correction_visualizer,
                            ideal_keypoints_3d=ideal_keypoints,
                            sample_rate=2
                        )
                        videos_created["3d_overlay"] = overlay_video_path
                except Exception:
                    pass
                
                # Create side-by-side comparison video
                try:
                    comparison_video_path = f"outputs/videos/{video_id}_comparison.mp4"
                    ideal_keypoints = get_ideal_keypoints(primary_activity)
                    if ideal_keypoints is not None:
                        video_overlay_corrector.create_side_by_side_video(
                            video_path,
                            comparison_video_path,
                            pose_estimator,
                            correction_visualizer,
                            ideal_keypoints_3d=ideal_keypoints,
                            sample_rate=2
                        )
                        videos_created["comparison"] = comparison_video_path
                except Exception:
                    pass
                
                # Create correction visualization video
                try:
                    correction_video_path = f"outputs/videos/{video_id}_correction.mp4"
                    ideal_keypoints = get_ideal_keypoints(primary_activity)
                    if ideal_keypoints is not None:
                        video_overlay_corrector.create_correction_video(
                            video_path,
                            correction_video_path,
                            pose_estimator,
                            correction_visualizer,
                            ideal_keypoints_3d=ideal_keypoints,
                            sample_rate=2
                        )
                        videos_created["correction"] = correction_video_path
                except Exception:
                    pass
                
                # Create animation video if ideal pose is available
                ideal_keypoints = get_ideal_keypoints(primary_activity)
                if ideal_keypoints is not None:
                    try:
                        animation_video_path = f"outputs/videos/{video_id}_animation.mp4"
                        video_overlay_corrector.create_correction_animation_video(
                            video_path,
                            animation_video_path,
                            pose_estimator,
                            correction_visualizer,
                            ideal_keypoints,
                            sample_rate=2,
                            animation_frames=10
                        )
                        videos_created["animation"] = animation_video_path
                    except Exception:
                        pass
                
            except Exception:
                pass

    except Exception:
        pass

    # Generate enhanced recommendations
    try:
        basic_recommendations = generate_recommendations(comprehensive_results, primary_activity)
        correction_recommendations_raw = correction_visualizer.generate_correction_recommendations(
            comprehensive_results, primary_activity
        )

        # Combine and format recommendations
        all_recommendations = basic_recommendations.copy()
        if isinstance(correction_recommendations_raw, list):
            for rec in correction_recommendations_raw:
                if isinstance(rec, dict):
                    priority = rec.get('priority', 'low')
                    priority_emoji = "🔥" if priority == 'high' else "⚡" if priority == 'medium' else "💡"
                    text = rec.get('text', 'No recommendation text')
                    score_info = f" (Score: {rec.get('score', 0):.2f})" if 'score' in rec else ""
                    all_recommendations.append(f"{priority_emoji} {text}{score_info}")
                elif isinstance(rec, str):
                    all_recommendations.append(rec)
                else:
                    all_recommendations.append(str(rec))
        elif isinstance(correction_recommendations_raw, str):
            all_recommendations.append(f"⚠️ {correction_recommendations_raw}")

    except Exception:
        all_recommendations = ["Could not generate recommendations due to analysis errors"]

    # Prepare final results
    try:
        overall_score = correction_visualizer.calculate_overall_score(comprehensive_results, primary_activity)
        improvement_areas = correction_visualizer.identify_improvement_areas(comprehensive_results, primary_activity)
        strengths = correction_visualizer.identify_strengths(comprehensive_results, primary_activity)
    except Exception:
        overall_score = 0.0
        improvement_areas = []
        strengths = []

    # Build final results dictionary
    final_results = {
        "video_info": {
            "file": video_path,
            "primary_activity": primary_activity,
            "frames_analyzed": len(frames_data),
            "valid_poses": stats.get('valid_poses', 0),
            "processing_stats": stats,
            "videos_created": videos_created if create_3d_videos else {}
        },
        "mechanics_analysis": comprehensive_results,
        "recommendations": all_recommendations,
        "correction_insights": {
            "overall_score": overall_score,
            "improvement_areas": improvement_areas,
            "strengths": strengths,
            "detailed_corrections": correction_recommendations_raw if 'correction_recommendations_raw' in locals() else []
        }
    }

    return final_results, primary_activity, video_path


def get_ideal_keypoints(activity_type: str) -> Optional[np.ndarray]:
    """
    Get ideal keypoints for the specified activity type.
    
    Args:
        activity_type: The type of cricket activity
        
    Returns:
        NumPy array of ideal keypoints or None if not available
    """
    if not activity_type or activity_type == "unknown":
        return None
        
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
        return None
        
    try:
        with open(ideal_pose_file, 'r') as f:
            ideal_pose_data = json.load(f)
            keypoints = ideal_pose_data.get('keypoints_3d')
            if keypoints:
                return np.array(keypoints)
            else:
                return None
    except Exception:
        return None


def generate_final_submission_report(all_results: Dict, summary_stats: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate comprehensive submission documentation.
    
    Args:
        all_results: Dictionary of all analysis results
        summary_stats: Summary statistics dictionary
        
    Returns:
        Tuple of (report_file_path, summary_file_path)
    """
    try:
        report_generator = SubmissionReportGenerator()
        analysis_data = {
            "analysis_summary": summary_stats,
            "detailed_results": all_results,
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "system_version": "Enhanced Cricket Analysis v2.1 with 3D Visualization"
            }
        }

        report_file = report_generator.generate_comprehensive_submission_report(
            analysis_data, output_dir="outputs"
        )

        if report_file:
            try:
                quick_summary = report_generator.generate_quick_summary(analysis_data)
                summary_file = f"outputs/submission_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(summary_file, 'w') as f:
                    json.dump(quick_summary, f, indent=2)
                
                return report_file, summary_file
            except Exception:
                return report_file, None

    except Exception:
        return None, None

    
def generate_recommendations(analysis_results: Dict, activity_type: str) -> List[str]:
    """
    Generate basic recommendations based on analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results
        activity_type: The primary activity type detected
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    try:
        # Get overall scores for each mechanic
        scores = {}
        for mechanic, results in analysis_results.items():
            if isinstance(results, dict) and 'scores' in results:
                scores_dict = results['scores']
                if isinstance(scores_dict, dict) and 'overall_score' in scores_dict:
                    scores[mechanic] = scores_dict['overall_score']
        
        # Generate recommendations based on scores
        batting_score = scores.get('batting_mechanics', 1.0)
        if batting_score < 0.7:
            recommendations.append("🏏 Focus on batting mechanics: work on stance and timing")
        
        bowling_score = scores.get('bowling_mechanics', 1.0)
        if bowling_score < 0.7:
            recommendations.append("🎳 Improve bowling technique: focus on run-up and release")
        
        fielding_score = scores.get('fielding_mechanics', 1.0)
        if fielding_score < 0.7:
            recommendations.append("🥅 Enhance fielding skills: practice anticipation and throwing")
        
        follow_through_score = scores.get('follow_through', 1.0)
        if follow_through_score < 0.7:
            recommendations.append("🔄 Work on follow-through: maintain balance and complete the motion")
        
        # Activity-specific recommendations
        if activity_type == "batting":
            if batting_score < 0.8:
                recommendations.append("🎯 Focus specifically on batting stance and head position")
        elif activity_type == "bowling":
            if bowling_score < 0.8:
                recommendations.append("🎯 Focus specifically on bowling action consistency")
        elif activity_type == "fielding":
            if fielding_score < 0.8:
                recommendations.append("🎯 Focus specifically on fielding positioning and reactions")
        
        # If no specific recommendations, add a general one
        if not recommendations:
            recommendations.append("👍 Good technique overall! Continue practicing to maintain form")
            
    except Exception:
        recommendations.append("Could not generate specific recommendations due to analysis errors")
    
    return recommendations


def main():
    """Main execution function."""
    print("🚀 FUTURESPORTLER ENHANCED CRICKET AI ANALYSIS")
    print("🏏 Analyzing: Batting • Bowling • Fielding • Follow-through")
    print("🔄 NEW: Advanced Correction Visuals with 3D Pose Comparisons and Video Output")
    print("🐛 FIXED: Boolean array errors and return value handling")
    print("✨ UPDATED: Analyzer output integration")
    print("="*70)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cricket Analysis with 3D Visualization')
    parser.add_argument('--input', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Path to output directory')
    parser.add_argument('--create_3d_videos', action='store_true', default=True, help='Create 3D visualization videos')
    parser.add_argument('--sample_rate', type=int, default=2, help='Process every nth frame (to speed up processing)')
    parser.add_argument('--max_frames', type=int, default=200, help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Initialize output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        print(f"📁 Output directory initialized: {args.output_dir}")
    except Exception as e:
        print(f"❌ Error creating output directories: {e}")
        return

    # Process single video if specified
    if args.input:
        if not os.path.exists(args.input):
            print(f"File not found: {args.input}")
            return
            
        video_id = os.path.splitext(os.path.basename(args.input))[0]
        results, activity, path = analyze_cricket_video(args.input, video_id, args.create_3d_videos)
        
        if results:
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(args.output_dir, f"analysis_{video_id}_{timestamp}.json")
            
            try:
                with open(output_file, "w") as f:
                    json.dump(convert_numpy_to_json_serializable(results), f, indent=2)
            except Exception as e:
                print(f"Error saving results: {e}")
            
            # Generate report
            summary_stats = {
                "total_videos": 1,
                "successful_analyses": 1,
                "failed_analyses": 0,
                "activity_breakdown": {activity: 1},
                "processing_errors": []
            }
            
            try:
                generate_final_submission_report({"video_1": results}, summary_stats)
            except Exception:
                pass
        else:
            print("Analysis failed")
            
        return

    # If no single video specified, process default videos
    video_files = [
        "data/Video-1.mp4",
        "data/Video-2.mp4",
        "data/Video-3.mp4",
        "data/Video-4.mp4",
        "data/Video-5.mp4"
    ]

    # Initialize tracking variables
    all_results = {}
    summary_stats = {
        "total_videos": 0,
        "successful_analyses": 0,
        "failed_analyses": 0,
        "activity_breakdown": {"batting": 0, "bowling": 0, "fielding": 0, "unknown": 0},
        "processing_errors": []
    }

    # Process each video
    for i, video in enumerate(video_files, 1):
        if not os.path.exists(video):
            print(f"File not found: {video}")
            summary_stats["failed_analyses"] += 1
            summary_stats["processing_errors"].append(f"File not found: {video}")
            continue

        summary_stats["total_videos"] += 1
        try:
            results, activity, path = analyze_cricket_video(video, f"video{i}", args.create_3d_videos)
            if results:
                all_results[f"video_{i}"] = results
                summary_stats["successful_analyses"] += 1
                summary_stats["activity_breakdown"][activity] += 1
            else:
                summary_stats["failed_analyses"] += 1
                summary_stats["processing_errors"].append(f"Analysis failed for: {video}")
                
        except Exception as e:
            print(f"Critical error processing {video}: {e}")
            summary_stats["failed_analyses"] += 1
            summary_stats["processing_errors"].append(f"Critical error in {video}: {str(e)}")

    # Save results and generate reports
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(args.output_dir, f"comprehensive_analysis_{timestamp}.json")
        
        try:
            success_rate = summary_stats["successful_analyses"] / summary_stats["total_videos"] if summary_stats["total_videos"] > 0 else 0
            
            comprehensive_data = {
                "analysis_metadata": {
                    "timestamp": timestamp,
                    "total_videos_processed": summary_stats["total_videos"],
                    "successful_analyses": summary_stats["successful_analyses"],
                    "failed_analyses": summary_stats["failed_analyses"],
                    "success_rate": success_rate,
                    "3d_videos_created": args.create_3d_videos,
                    "max_frames_per_video": args.max_frames,
                    "sample_rate": args.sample_rate
                },
                "analysis_summary": summary_stats,
                "detailed_results": all_results
            }
            
            with open(output_file, "w") as f:
                json.dump(convert_numpy_to_json_serializable(comprehensive_data), f, indent=2)
        except Exception:
            pass

        # Generate final reports
        try:
            generate_final_submission_report(all_results, summary_stats)
        except Exception:
            pass
    else:
        print("No successful analyses to save")

    # Print final summary
    print(f"Videos processed: {summary_stats['total_videos']}")
    print(f"Successful analyses: {summary_stats['successful_analyses']}")
    print(f"Failed analyses: {summary_stats['failed_analyses']}")
    
    if summary_stats['total_videos'] > 0:
        success_rate = summary_stats['successful_analyses'] / summary_stats['total_videos'] * 100
        print(f"Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()