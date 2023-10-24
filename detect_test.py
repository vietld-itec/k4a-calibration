import cv2
import numpy as np
import eigen3 as e3
from april_tag import AprilTagDetector

# Load the color image from frames[camera_index]
color_image = frames[camera_index].ColorImage
color_height, color_width = frames[camera_index].ColorHeight, frames[camera_index].ColorWidth

# Convert the color image to grayscale
gray = cv2.cvtColor(color_image, cv2.COLOR_BGRA2GRAY)

# Create an image_u8_t struct
orig = e3.ImageU8(color_width, color_height, color_width, gray.data.tobytes())

# Detect AprilTags
detections = AprilTagDetector.detect(orig)
print(f"Detected {len(detections)} fiducial markers")
found = False

# Accessing CameraCalibration
calibration = frames[camera_index].Calibration

for det in detections:
    print(f"Camera {camera_index} detected marker ID: {det.id}")
    if det.id != 0:
        print(f"Camera {camera_index} detected incorrect marker #{det.id}")
        continue

    info = e3.AprilTagDetectionInfo()
    info.det = det
    info.cx = calibration.Color.cx  # pixels
    info.cy = calibration.Color.cy
    info.fx = calibration.Color.fx  # mm
    info.fy = calibration.Color.fy
    info.tagsize = 0.22  # in meters

    pose = e3.AprilTagPose()
    err = e3.estimate_tag_pose(info, pose)

    tr = pose.R.data
    tt = pose.t.data

    print(f"Object-space error = {err}")
    print("R = [")
    for i in range(3):
        print(f"  {tr[i*3]:.4f}, {tr[i*3+1]:.4f}, {tr[i*3+2]:.4f}")
    print("]")
    print(f"t = [{tt[0]:.4f}, {tt[1]:.4f}, {tt[2]:.4f}]")

    transform = np.identity(4, dtype=np.float32)
    rotmat = np.zeros((3, 3), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            transform[row, col] = tr[row * 3 + col]
            rotmat[row, col] = tr[row * 3 + col]
    for row in range(3):
        transform[row, 3] = tt[row]
    for col in range(3):
        transform[3, col] = 0.0
    transform[3, 3] = 1.0

    # cube_transform code (uncomment and adjust as needed)

    # tag_poses[camera_index] = np.dot(transform, np.linalg.inv(cube_transform))
    # tag_pose = np.dot(tag_poses[0], np.linalg.inv(tag_poses[camera_index]))
    # current_transform[camera_index] = tag_pose

    found = True
