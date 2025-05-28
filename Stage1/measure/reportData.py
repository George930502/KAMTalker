import os
import cv2
import random
import shutil

def get_video_files(folder):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return [os.path.join(folder, f) for f in os.listdir(folder) 
            if f.lower().endswith(video_extensions)]

def get_random_video(video_folder):
    videos = get_video_files(video_folder)
    if not videos:
        raise ValueError("No videos found in the folder!")
    return random.choice(videos)

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def extract_random_frame(video_path):
    total_frames = get_total_frames(video_path)
    if total_frames <= 0:
        raise ValueError("Unable to get total frames from the video.")
    random_index = random.randint(0, total_frames - 1)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Unable to extract frame {random_index}")
    return frame

#####################################
# Module 1:
# Randomly select a video, extract two distinct frames,
# and save them in the output folder as "source.jpg" and "driving.jpg".
#####################################
def module1(video_folder, output_folder):
    video_path = get_random_video(video_folder)
    total_frames = get_total_frames(video_path)
    if total_frames < 2:
        raise ValueError("The video does not have enough frames to extract two frames.")
    
    # Randomly choose two different frame indices
    frame_indices = random.sample(range(total_frames), 2)
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Unable to extract frame {idx}")
        frames.append(frame)
    cap.release()
    
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the two frames as "source.jpg" and "driving.jpg"
    output_path1 = os.path.join(output_folder, "source.jpg")
    output_path2 = os.path.join(output_folder, "driving.jpg")
    
    cv2.imwrite(output_path1, frames[0])
    cv2.imwrite(output_path2, frames[1])
    
    print(f"Module 1: Extracted two frames from {video_path} and saved to:")
    print(f"  {output_path1}")
    print(f"  {output_path2}")

#####################################
# Module 2:
# Randomly select a video, copy it to the output folder as "driving.mp4",
# and extract a random frame from the video, saving it as "source.jpg" in the same folder.
#####################################
def module2(video_folder, output_folder):
    video_path = get_random_video(video_folder)
    
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy the video file into the output folder with a fixed name
    dest_video_path = os.path.join(output_folder, "driving.mp4")
    shutil.copy(video_path, dest_video_path)
    
    # Extract a random frame and save it as "source.jpg"
    frame = extract_random_frame(video_path)
    output_frame_path = os.path.join(output_folder, "source.jpg")
    cv2.imwrite(output_frame_path, frame)
    
    print(f"Module 2: Copied video {video_path} to {dest_video_path}")
    print(f"         Extracted frame saved to {output_frame_path}")

#####################################
# Module 3:
# Randomly select two different videos, extract one frame from each,
# and save them in the output folder as "source.jpg" (from the first video)
# and "driving.jpg" (from the second video).
#####################################
def module3(video_folder, output_folder):
    videos = get_video_files(video_folder)
    if len(videos) < 2:
        raise ValueError("Not enough videos in the folder (need at least 2).")
    
    selected_videos = random.sample(videos, 2)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract frame from the first video and save as "source.jpg"
    frame1 = extract_random_frame(selected_videos[0])
    output_path1 = os.path.join(output_folder, "source.jpg")
    cv2.imwrite(output_path1, frame1)
    print(f"Module 3: Extracted frame from {selected_videos[0]} saved to {output_path1}")
    
    # Extract frame from the second video and save as "driving.jpg"
    frame2 = extract_random_frame(selected_videos[1])
    output_path2 = os.path.join(output_folder, "driving.jpg")
    cv2.imwrite(output_path2, frame2)
    print(f"Module 3: Extracted frame from {selected_videos[1]} saved to {output_path2}")

#####################################
# Module 4:
# Randomly select two different videos.
# From the first video, extract a random frame and save it as "source.jpg".
# From the second video, copy the complete video file and save it as "driving.mp4".
#####################################
def module4(video_folder, output_folder):
    videos = get_video_files(video_folder)
    if len(videos) < 2:
        raise ValueError("Not enough videos in the folder (need at least 2).")
    
    selected_videos = random.sample(videos, 2)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract a random frame from the first video and save as "source.jpg"
    frame = extract_random_frame(selected_videos[0])
    frame_path = os.path.join(output_folder, "source.jpg")
    cv2.imwrite(frame_path, frame)
    
    # Copy the complete second video and save as "driving.mp4"
    dest_video_path = os.path.join(output_folder, "driving.mp4")
    shutil.copy(selected_videos[1], dest_video_path)
    
    print(f"Module 4: Extracted frame from {selected_videos[0]} saved to {frame_path}")
    print(f"          Copied complete video from {selected_videos[1]} to {dest_video_path}")

#####################################
# Main function to iterate each module N times.
# For each module, the results are saved into separate directories
# such as "results/{module_number}/test{iteration}".
#####################################
if __name__ == "__main__":
    # Adjust the video folder path as needed
    video_folder = "/root/video_generation/video_enhancement/GFPGAN/pure_talking_faces/test"
    
    # Number of iterations for each module
    N = 500
    
    # Iterate Module 1 N times and store results in "results/1/test{iteration}"
    for i in range(1, N + 1):
        output_folder = os.path.join("results", "1", f"test{i}")
        module1(video_folder, output_folder)
    
    # Iterate Module 2 N times and store results in "results/2/test{iteration}"
    for i in range(1, N + 1):
        output_folder = os.path.join("results", "2", f"test{i}")
        module2(video_folder, output_folder)
    
    # Iterate Module 3 N times and store results in "results/3/test{iteration}"
    for i in range(1, N + 1):
        output_folder = os.path.join("results", "3", f"test{i}")
        module3(video_folder, output_folder)
    
    # Iterate Module 4 N times and store results in "results/4/test{iteration}"
    for i in range(1, N + 1):
        output_folder = os.path.join("results", "4", f"test{i}")
        module4(video_folder, output_folder)
