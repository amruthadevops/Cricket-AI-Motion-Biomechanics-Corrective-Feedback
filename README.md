
# FutureSportler Cricket AI — Motion, Biomechanics & Corrective Feedback

---

FutureSportler Cricket AI is a video-based AI system that provides **3D cricket motion analysis**, **biomechanics scoring**, and **coach-style corrective feedback** for batting, bowling, and fielding.

---
## Demo Video
[🎥 Watch the demo on Google Drive](https://drive.google.com/drive/folders/16h23KP6vGn1OOYR-RKRitXlbHM_RaG1E?usp=drive_link)
---

## 🎯 Problem-Solving Aim

This project aims to address two key practical problems:

1.  To make **biomechanical feedback accessible to coaches** without the need for expensive motion-capture setups.
2.  To provide **reproducible, interpretable scoring** for movement quality that can be extended into a coaching pipeline or product.

The approach thoughtfully balances **accuracy, interpretability, and runtime performance** (using CPU-capable models and rule-based metrics). This enables quick iteration and production of high-quality demonstration outputs suitable for interviews and prototypes.

---

### Recorded Outputs (examples in `outputs/`)

* **Final Annotated Video:** `outputs/annotated_final_video_1.mp4` — The original video with overlays and textual feedback.
* **Bat Swing Trajectory & Timing Plot:** `outputs/swing_plot_video1.png` — A 2D plot illustrating bat-wrist path and key timing markers.
* **Interactive 3D Skeleton Viewer:** `outputs/skeleton_video1.html` — An interactive 3D viewer (open in your browser) to rotate and inspect the pose.
* **Actual vs. Ideal 3D Pose Comparison:** `outputs/pose_comparison_video1.html` — Visualize the actual pose against an ideal reference skeleton for visual correction.
* **Interactive Per-Frame Dashboard:** `outputs/correction_dashboard_video1.html` — A lightweight dashboard with charts for timing, angles, confidence, and a per-frame list of corrections.
* **Structured Feedback:** `outputs/corrections_<activity>_Video-*.json` — Per-video structured feedback with details like frame, issue, severity, and suggested fixes.
* **Full Aggregated Analytics:** `outputs/comprehensive_analysis_*.json` — A global summary across all processed videos.
* **Auto-generated Submission Documentation:** `submission_report_*.md` & `submission_summary_*.json`

---

## 📝 Table of Contents

* [Project Overview](#-project-overview)
* [Problem-Solving Aim](#-problem-solving-aim)
* [Approach](#-approach-high-level-pipeline)
* [Tools & Libraries](#-tools--libraries)
* [Key Features Detected & Reasoning](#-key-features-detected--reasoning-details)
    * [Batting](#batting-what-we-detect)
    * [Bowling](#bowling-what-we-detect)
    * [Fielding](#fielding-what-we-detect)
* [How to Run (Reproduce)](#-how-to-run-reproduce)
* [Repository Structure](#-repository-structure-what-to-commit)
* [Limitations & Assumptions](#-limitations--assumptions)
* [Next Steps / Future Work](#-next-steps--future-work)
* [Credits & License](#-credits--license)
* [Contact](#-contact)

---

## 💡 Project Overview

This project showcases a complete pipeline that processes **single-view cricket match/practice video clips**. It performs **2D pose estimation**, lifts it to a **3D representation**, extracts **biomechanical features and heuristics** for batting, bowling, and fielding, generates **per-frame and per-action scores**, and produces **coach-style corrective feedback with polished visualizations**.

The primary goal is to generate outputs that are:

* **Actionable**: Providing clear corrective tips for coaches and players.
* **Interpretable**: Displaying angles, timing windows, and simple heuristics instead of opaque "black-box" predictions.
* **Reproducible**: Capable of running on a standard CPU-based environment using MediaPipe/TFLite.

---

## 📊 Video Outputs (What's Included & How to Read Them)

All generated outputs are saved to the `outputs/` directory after processing a video clip. The key files for a GitHub release include:

* `annotated_final_video_<id>.mp4`: The final merged video, including original frames, 3D/2D overlays, and text annotations explaining issues and suggestions.
* `swing_plot_video<id>.png`: A 2D plot visualizing the bat-wrist path and important timing markers (contact window, trigger point).
* `skeleton_video<id>.html`: An interactive 3D skeleton viewer (exported using Three.js); you can rotate and inspect the pose directly in your browser.
* `pose_comparison_video<id>.html`: Displays the actual skeleton pose versus an "ideal" reference skeleton, highly useful for visual correction.
* `correction_dashboard_video<id>.html`: A lightweight dashboard featuring charts (timing, angles, confidence, and a per-frame list of corrections).
* `corrections_<activity>_Video-<id>.json`: Structured corrections containing fields such as frame, issue, severity, and `suggested_fix`.
* `comprehensive_analysis_<timestamp>.json`: A global summary of analytics across all processed videos.

### How Reviewers Should Inspect:

1.  **Watch** one `annotated_final_video*.mp4` to get an instant impression of the system's capabilities.
2.  **Open** `pose_comparison*.html` to interact with the 3D skeletons and visually verify the corrections.
3.  **Inspect** the JSON files for precise numeric metrics if needed.

---

## ⚙️ Approach (High-Level Pipeline)

The system follows this high-level pipeline:

1.  **Video Input**: Frames are read using OpenCV and resized to a consistent resolution.
2.  **2D Pose Estimation**: MediaPipe Pose (or OpenPose) is used to obtain 33 keypoints per frame, along with their `(x, y, z, visibility)` coordinates.
3.  **Filtering & Smoothing**: Temporal smoothing and interpolation are applied to handle missed detections and provide more consistent data.
4.  **2D → 3D Lifting / Visualization**: Simple lifting and re-projection techniques are used to display a 3D skeleton in `skeleton_*.html` (using Three.js or Plotly for interactivity).
5.  **Feature Extraction / Heuristics**:
    * **Bat Angle**: Calculated as the vector between the wrist and the inferred bat-tip direction (derived from wrist and elbow orientation).
    * **Trigger Movement**: Detection of early foot/hip shifts before the swing via derivatives of keypoint positions.
    * **Release Detection**: Identified by the wrist velocity peak during bowling delivery.
    * **Dive Detection**: Recognized by sustained large horizontal body velocity combined with torso orientation.
    * **Torque Estimation**: Measured as the difference between shoulder rotation and hip rotation angles.
6.  **Scoring/Rule-based Evaluation**: Heuristics are normalized to 0–1 scores and aggregated into sub-scores (batting, bowling, fielding, follow-through).
7.  **Correction Generation**: Threshold violations are converted into human-readable corrections with associated severity levels.
8.  **Visual Overlays**: Skeletons, angle arcs, arrows, and `cv2.putText()` guidance are drawn onto the frames to create the final annotated `.mp4`.
9.  **Reports**: JSON and HTML dashboards are saved for more in-depth review.

---

## 🛠️ Tools & Libraries

The project is built using:

* **Python 3.10+**
* **MediaPipe**: For fast 2D pose keypoints (CPU-friendly).
* **OpenCV**: For video I/O and drawing overlays.
* **NumPy / SciPy**: For mathematical operations and smoothing.
* **Matplotlib / Plotly**: For generating plots and small charts.
* **moviepy (optional)**: For video composition.
* **three.js / pythreejs / Plotly**: For interactive 3D skeleton export to HTML.
* **TFLite delegate**: Used where available to speed up low-level inference.


## requirements

```
mediapipe==0.10.0
opencv-python==4.7.0
numpy
matplotlib
plotly
moviepy
scipy
```

## ✨ Key Features Detected & Reasoning (Details)

This section concisely explains what the system detects, how it detects it, and why it matters.

### Batting (What We Detect)

* **Stance & Alignment**: Compares shoulder line versus hip line to assess side-on vs. front-on stance, which influences shot selection.
    * **Reasoning**: Proper hip-shoulder alignment ensures a stable hitting base.
* **Trigger Movement**: Measures lateral hip/foot displacement in the 150–300ms window before the swing.
    * **Why**: Early or late triggers can significantly affect shot timing and control.
* **Bat Angle**: The angle between the inferred bat vector and vertical/horizontal axes.
    * **Why**: A steep bat face often leads to top-edge or high-ball trajectories, while an open face can cause wide shots.
* **Shot-type Heuristics**: Drive, pull, cut, and defensive shots are inferred through a combination of bat vector direction, hip rotation, and front/back foot weight.
    * **Why**: This helps in producing targeted coaching feedback (e.g., "adjust foot position for drives").

### Bowling (What We Detect)

* **Run-up Consistency**: Analyzes stride length and variance across approach frames.
    * **Why**: A stable run-up reduces release variability.
* **Load-up Segmentation**: Identifies gather → front-foot plant → arm-cocking frames using hip/shoulder markers.
    * **Why**: Crucial for assessing pace generation and potential injury risk.
* **Release Dynamics**: Evaluates wrist velocity, elbow extension speed, and release angle.
    * **Reasoning**: The release point determines line/length, and wrist snap affects swing/spin.
* **Shoulder-Hip Torque**: The angular difference between shoulders and hips just before release.
    * **Why**: High separation often correlates with higher pace and efficient power transfer.

### Fielding (What We Detect)

* **Anticipation / Reaction**: Measures the time between a ball direction change (or ball arrival frame) and the first significant movement of hands/feet.
* **Dive Mechanics**: Analyzes horizontal body velocity, arm extension, and impact posture during a dive.
* **Throwing Angle**: Examines the shoulder-elbow-wrist orientation at the point of release during a throw.

---



## 🏃‍♀️ How to Run (Reproduce)

### Clone & Setup

```bash
git clone [https://github.com/](https://github.com/)<your-username>/futuresportler-cricket-ai.git
cd futuresportler-cricket-ai
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
py main.py
```
## Limitations & Assumptions

* **Single-view:** Accuracy is limited compared to multi-view stereo or MoCap systems. Torque and absolute depth estimations are approximations.

* **Camera Angle & Occlusion:** Overhead or extreme camera angles can degrade pose estimation quality.

* **No Heavy ML Training:** The system relies on rules and heuristics for classification; adding a small classifier could improve shot-type accuracy.

* **Frame Rate Dependency:** Results assume a 50–60 FPS input; downsampling will alter timing thresholds.

## 🚀 Next Steps / Future Work

- Integrate a small supervised classifier for shot-type (train on labeled clips).

- Improve 3D lifting (explore multi-view or learned depth lifting models).

- Develop a lightweight web UI (e.g., using Streamlit) for interactive review and coach notes.

- Add unit tests & CI to validate reproducibility and ensure code quality.


## 📜 Credits & License

- [@Amrutha](https:https://www.linkedin.com/in/c-amrutha/)

