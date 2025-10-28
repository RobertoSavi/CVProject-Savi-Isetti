# 🏀 3D Human Pose Reconstruction and Motion Capture Alignment  

### *Computer Vision Project — University of Trento (A.Y. 2024–2025)*  
**Authors:** Jacopo Isetti, Roberto Savi  
**Course:** Computer Vision  

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


---

## ⚙️ Installation & Setup  

### Prerequisites
- Python ≥ 3.10  
- Install required libraries:
  ```bash
  pip install numpy opencv-python matplotlib scipy mediapipe torch vedo

