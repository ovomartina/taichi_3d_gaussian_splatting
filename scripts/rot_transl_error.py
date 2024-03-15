import os
import numpy as np
import matplotlib.pyplot as plt

# Define the target vector
target_vetor_t = np.array([0.3837, -0.0831, 0.4156])
target_vetor_q = np.array([0.6188, 0.5829, -0.3611, -0.3834])
# Initialize a list to store errors
errors_t = []
errors_q = []
# Define the folder path
folder_path = "/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/epochs"

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith("_t.txt"):
        # Load the content of the file as a numpy array
        file_path = os.path.join(folder_path, filename)
        loaded_array = np.loadtxt(file_path).reshape(3, 1)
        
        # Compute the norm between the loaded array and the target vector
        error = np.linalg.norm(loaded_array - target_vetor_t)
        
        # Add the error to the list
        errors_t.append(error)
    if filename.endswith("_q.txt"):
        # Load the content of the file as a numpy array
        file_path = os.path.join(folder_path, filename)
        loaded_array = np.loadtxt(file_path).reshape(4, 1)
        
        # Compute the norm between the loaded array and the target vector
        error = np.linalg.norm(loaded_array - target_vetor_q)
        
        # Add the error to the list
        errors_q.append(error)

# Plot the errors_t
plt.plot(errors_t)
plt.xlabel("File Index")
plt.ylabel("Error")
plt.title("Error for Files Ending with '_t.txt'")
plt.show()

plt.plot(errors_q)
plt.plot(errors_t)
plt.xlabel("File Index")
plt.ylabel("Error")
plt.title("Rotational and translational error")
plt.savefig("/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/scripts/3dgs_playground_output/epochs/rot_trasl_error.png")
plt.show()