#!/usr/bin/env python

import os
import cv2
import random
import matplotlib.pyplot as plt

# Define paths for CAUCAFall falls and non-falls folders
FALLS_PATH = ["/Users/gwen/ITSS_Project/processed_data/CAUCAFall/falls",
              "/Users/gwen/ITSS_Project/processed_data/GMDCSA24/falls"
]
NON_FALLS_PATH = ["/Users/gwen/ITSS_Project/processed_data/CAUCAFall/non-falls",
                  "/Users/gwen/ITSS_Project/processed_data/GMDCSA24/non-falls"
]

def get_video_frame_counts(category_path):
    """
    For a given category path (falls or non-falls), list each video folder
    and count the number of JPEG frames inside.
    """
    video_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
    frame_counts = {}
    for video_folder in video_folders:
        folder_path = os.path.join(category_path, video_folder)
        # Assuming frames are saved as .jpg files following the naming convention
        frame_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        frame_counts[video_folder] = len(frame_files)
    return frame_counts

def plot_frame_count_distribution(frame_counts, category):
    """
    Plot a histogram of the frame counts for a given category.
    """
    counts = list(frame_counts.values())
    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=10, edgecolor='black')
    plt.title(f'Frame Count Distribution for {category} Videos')
    plt.xlabel('Number of Frames per Video')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def display_random_sample(category_path, category):
    """
    Display a random frame from a random video within the given category.
    """
    video_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
    if not video_folders:
        print(f"No video folders found in {category} category.")
        return

    # Pick a random video folder
    random_video = random.choice(video_folders)
    folder_path = os.path.join(category_path, random_video)
    
    frame_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    if not frame_files:
        print(f"No frames found in video folder: {random_video}")
        return

    # Pick a random frame from the selected video folder
    random_frame = random.choice(frame_files)
    frame_path = os.path.join(folder_path, random_frame)
    
    # Read and display the image
    image = cv2.imread(frame_path)
    if image is None:
        print(f"Failed to load image: {frame_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f'Random Frame from {random_video} ({category})')
    plt.axis('off')
    plt.show()

def main():
    # Process Falls category
    falls_counts = get_video_frame_counts(FALLS_PATH)
    print("Falls Video Frame Counts:")
    for video, count in falls_counts.items():
        print(f"{video}: {count} frames")
    
    # Process Non-Falls category
    non_falls_counts = get_video_frame_counts(NON_FALLS_PATH)
    print("\nNon-Falls Video Frame Counts:")
    for video, count in non_falls_counts.items():
        print(f"{video}: {count} frames")
    
    # Plot distribution histograms
    plot_frame_count_distribution(falls_counts, "Falls")
    plot_frame_count_distribution(non_falls_counts, "Non-Falls")
    
    # Display random sample frames for visual verification
    display_random_sample(FALLS_PATH, "Falls")
    display_random_sample(NON_FALLS_PATH, "Non-Falls")

if __name__ == "__main__":
    main()
