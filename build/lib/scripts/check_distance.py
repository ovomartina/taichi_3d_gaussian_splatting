import json
import numpy as np

# Load the JSON file containing transform matrices
with open('/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/data/replica_colmap/room_1_high_quality_50_frames_test/val.json', 'r') as file:
    transforms_data = json.load(file)

# Convert JSON data to numpy arrays
transforms = [np.array(transform["T_pointcloud_camera"]) for transform in transforms_data]

# Compute Euclidean distance between consecutive transforms
distances = []
for i in range(1, len(transforms)):
    distance = np.linalg.norm(transforms[i] - transforms[i-1])
    distances.append(distance)

# Print the distances
for i, distance in enumerate(distances, start=1):
    print(f"Distance between transform {i} and transform {i+1}: {distance}")