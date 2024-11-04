import os
import cv2


video_dirs = {
    "train_abnormal": "videos/train/abnormal/",
    "train_normal": "videos/train/normal/",
    "test_abnormal": "videos/test/abnormal/",
    "test_normal": "videos/test/normal/",
    "val_abnormal": "videos/validation/abnormal/",
    "val_normal": "videos/validation/normal/",
}


frame_dirs = {
    "train_abnormal": "frames/train/abnormal/",
    "train_normal": "frames/train/normal/",
    "test_abnormal": "frames/test/abnormal/",
    "test_normal": "frames/test/normal/",
    "val_abnormal": "frames/validation/abnormal/",
    "val_normal": "frames/validation/normal/",
}



for folder in frame_dirs.values():
    os.makedirs(folder, exist_ok=True)



def extract_frames_from_video(video_path, frame_output_dir, frame_count_file):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_output_dir = os.path.join(frame_output_dir, video_name)
    os.makedirs(frame_output_dir, exist_ok=True)


    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"can not open video: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        frame_filename = os.path.join(frame_output_dir, f"{frame_count:06d}.jpg")


        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"finish: {video_name}, frames count: {frame_count}")


    with open(frame_count_file, "a") as f:
        f.write(f"{video_name}: {frame_count} frames\n")


for key, video_dir in video_dirs.items():
    frame_dir = frame_dirs[key]

    frame_count_file = os.path.join(frame_dir, "frame_counts.txt")


    with open(frame_count_file, "w") as f:
        f.write("video_name: frame_num\n")

    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
          
                extract_frames_from_video(video_path, frame_dir, frame_count_file)

print("finish processing!")
