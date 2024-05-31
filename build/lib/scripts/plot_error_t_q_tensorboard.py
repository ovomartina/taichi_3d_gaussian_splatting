import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_scalars_from_events(event_files, tags):
    data = {tag: [] for tag in tags}
    steps = {tag: [] for tag in tags}

    for event_file in event_files:
        try:
            for summary in tf.compat.v1.train.summary_iterator(event_file):
                for value in summary.summary.value:
                    if value.tag in tags:
                        data[value.tag].append(value.simple_value)
                        steps[value.tag].append(summary.step)
        except Exception as e:
            print(f"Error reading {event_file}: {e}")

    for tag in tags:
        data[tag] = np.array(data[tag])
        steps[tag] = np.array(steps[tag])
        if len(data[tag]) == 0:
            print(f"No data found for tag {tag} in event files.")
        else:
            sorted_indices = np.argsort(steps[tag])
            data[tag] = data[tag][sorted_indices]
            steps[tag] = steps[tag][sorted_indices]

    return data, steps

def plot_data(data, steps, tags):
    plt.figure(figsize=(12, 6))
    steps_q =0
    steps_t = 0
    data_q=0
    data_t=0
    for tag in tags:
        if len(data[tag]) > 0:
            print(data[tag][0])
            print(data[tag][-1])
            if tag == 'train/total_error_q':
                steps_q = steps[tag]
                data_q = data[tag]
                # plt.plot(steps[tag], data[tag], label=tag, color="orange")
            else:
                steps_t = steps[tag]
                data_t = data[tag]
                # plt.plot(steps[tag], data[tag], label=tag)
        else:
            print(f"No data to plot for tag {tag}")

    plt.figure(figsize=(10, 6))
    plt.xlabel(f"Epoch", fontsize="x-large")
    plt.plot(steps_t, data_t,label='Translation Error ')
    plt.legend(loc='upper right', fontsize="x-large")
    plt.ylabel('Translation [m]', fontsize="x-large")
    plt.twinx()
    plt.ylabel(f"Rotation [rad]", fontsize="x-large")
    plt.plot(steps_q, data_q, color='orange', label='Rotation Error')
    plt.title(f"Translational  and Rotational Error", fontsize="x-large")
    plt.legend(loc=(0.7, 0.8), fontsize="x-large")
    plt.grid(True)
    plt.savefig('logs/replica_colmap/room_1_high_quality_500_frames_bundle_adjustment_continuous_old_1/plot.png', dpi=600)

# Directory containing the TensorBoard event files
event_dir = 'logs/replica_colmap/room_1_high_quality_500_frames_bundle_adjustment_continuous_old_1'
event_files = [os.path.join(event_dir, f) for f in os.listdir(event_dir) if f.startswith('events.out.tfevents')]

# Tags you want to retrieve
tags = ['train/total_error_t', 'train/total_error_q']

# Load scalar data from event files
data, steps = load_scalars_from_events(event_files, tags)

# Plot the data
plot_data(data, steps, tags)