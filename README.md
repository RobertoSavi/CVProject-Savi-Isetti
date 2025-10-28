# 🏀 3D Human Pose Reconstruction and Motion Capture Alignment  

### *Computer Vision Project — University of Trento (A.Y. 2024–2025)*  
**Authors:** Jacopo Isetti, Roberto Savi  
**Course:** Computer Vision  
**Tutor:** Niccolò Bisagno — *University of Trento*  

---

## 📖 Overview  

This project focuses on **3D human pose reconstruction** from a **multi-view basketball sequence**, using synchronized RGB cameras and comparing the reconstructed 3D skeleton with **motion capture (MoCap)** ground truth data.  
It was developed as part of the *Computer Vision* course at the University of Trento.

The main objectives were to:

1. **Annotate** the 2D skeletons of a player from multiple calibrated cameras.  
2. **Triangulate** the 3D joint positions from multi-view correspondences.  
3. **Visualize and evaluate** the 3D reconstruction accuracy against 2D ground truth.  
4. **Align** the reconstructed sequence with MoCap data and compare 3D accuracy.  

---

## 🎯 Project Goals  

> Estimate the player’s 3D pose using the multiview camera setup at Sanbàpolis and evaluate accuracy by comparing against motion capture data.

| Step | Task | Description |
|------|------|-------------|
| **1** | 2D Annotation | Manual annotation of the player’s joints using Roboflow. |
| **2** | 3D Triangulation | Reconstruct 3D skeletons from 2D annotations (Steps 2.1–2.4). |
| **3** | MoCap Alignment | Align RGB and MoCap sequences and evaluate triangulation accuracy (Steps 3.1–3.3). |
| **3a** | Evaluation | Quantify error (MPJPE, MSE) between 3D triangulations and MoCap ground truth. |

---

## 🧩 Repository Structure  
📦 ComputerVision
├── resources/ # Input data and calibrations
│ ├── annotations/ # 2D keypoint labels (YOLO format)
│ ├── cameras/ # Camera calibration and metadata
│ ├── mocap/ # Motion capture .mat file (Nick_2.mat)
│ └── videos/ # Multi-view videos of the basketball sequence
│
├── src/ # Source code (main processing pipeline)
│ ├── step2_1.py # Rectify frames and annotations
│ ├── step2_2.py # Triangulate 3D joints
│ ├── step2_3.py # Visualize 3D skeletons and create animations
│ ├── step2_4.py # Reproject 3D joints and compute 2D errors
│ ├── step3_1.py # MoCap peak detection and temporal alignment
│ ├── step3_2.py # MoCap-to-video overlay visualization
│ ├── step3_3.py # 3D alignment (PnP + rigid transform averaging)
│ └── step3_MoCap_plot.py # MoCap animation generator
│
├── results/ # Generated outputs
│ ├── triang_3d/ # Triangulated 3D points
│ ├── rectified_videos/ # Rectified input videos
│ ├── triang_overlays/ # 2D projections overlays
│ ├── mocap_overlays/ # MoCap projection overlays
│ ├── triang_error_plots/ # 2D triangulation error plots
│ ├── mocap_triang_3d/ # 3D comparisons MoCap vs Triangulation
│ └── video_mocap_time_aligned.mp4
│
└── README.md

---

## ⚙️ Installation & Setup  

### Prerequisites
- Python ≥ 3.10  
- Install required libraries:
  ```bash
  pip install numpy opencv-python matplotlib scipy mediapipe torch vedo

