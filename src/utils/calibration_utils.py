import numpy as np
import json
import cv2

# Load camera calibration matrix and distortion coefficients from a JSON file
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    tvecs = np.array(calib["tvecs"], dtype=np.float32).reshape(3, 1)
    rvecs = np.array(calib["rvecs"], dtype=np.float32).reshape(3, 1)
    
    K_rect = None
    if "K_rect" in calib:
        K_rect = np.array(calib["K_rect"], dtype=np.float32)
        
    P = None
    if "P" in calib:
        P = np.array(calib["P"], dtype=np.float32)
          
    return mtx, dist, tvecs, rvecs, K_rect, P

def compute_projection_matrix(mtx, tvecs, rvecs):
    K = mtx
    # Convert rvec in R
    R, _ = cv2.Rodrigues(rvecs)
    # Create the matrix [R | t]
    Rt = np.hstack((R, tvecs))
    # Projection matrix
    P = K @ Rt  
    return P