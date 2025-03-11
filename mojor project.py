# %%
!pip install ultralytics

# %%
# Import all necessary libraries
from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from tqdm.notebook import tqdm

# %%
pip install kaggle


# %%
import torch
print(torch.cuda.is_available())  # Should return False (No GPU)
print(torch.device("cpu"))  # Should return 'cpu'




# %%
pip install ultralytics opencv-python numpy matplotlib

# %%
import ultralytics
print(ultralytics.__version__)  # Should print the installed version

# %%
!pip install ultralytics opencv-python numpy matplotlib tqdm torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# %%
import ultralytics
import torch

print("Ultralytics Version:", ultralytics.__version__)  # Should print YOLO version
print("PyTorch Version:", torch.__version__)  # Should print PyTorch version
print("CUDA Available:", torch.cuda.is_available())  # Should return False (since using CPU)


# %%
import os

# Set the dataset directory (update this path if needed)
dataset_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS_Dataset\Abuse"

# Check if the dataset exists
if os.path.exists(dataset_path):
    print("Dataset found at:", dataset_path)
else:
    print("Dataset path is incorrect. Please check!")


# %%
import os

# Base directory where the dataset is supposed to be
base_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project"

# List all files and folders inside it
print("📂 Checking files in:", base_path)
if os.path.exists(base_path):
    print("\n".join(os.listdir(base_path)))
else:
    print("❌ Path does not exist. Check if the folder is correct.")


# %%
import os

dataset_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset"

print(f"📂 Checking files in: {dataset_path}")
if os.path.exists(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        print(f"\n📂 Folder: {root}")
        print(f"📁 Subfolders: {dirs}")
        print(f"📄 Files: {files}")
else:
    print("❌ Path does not exist. Check if the folder is correct.")


# %%
import os

dataset_path = "DCSASS"  # Change this to the full path if needed

if not os.path.exists(dataset_path):
    print(f"❌ Error: Dataset folder '{dataset_path}' not found.")
else:
    print(f"✅ Dataset found: {dataset_path}")


# %%
import os

dataset_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset"

if os.path.exists(dataset_path):
    print(f"✅ Found dataset at: {dataset_path}")
    print("Subfolders inside:", os.listdir(dataset_path))
else:
    print(f"❌ Error: Dataset folder '{dataset_path}' not found. Check the path.")


# %%
import cv2

video_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset\Abuse\Abuse001_x264.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ OpenCV cannot open the video file. Check if it's accessible.")
else:
    print("✅ OpenCV successfully opened the video.")

cap.release()


# %%
import os

dataset_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset"

print(f"📂 Checking files in: {dataset_path}")
if os.path.exists(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        print(f"\n📂 Folder: {root}")
        print(f"📁 Subfolders: {dirs}")
        print(f"📄 Files: {files}")
else:
    print("❌ Path does not exist. Check if the folder is correct.")


# %%
import os

# Define the dataset path
dataset_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset"

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    print("❌ Path does not exist. Check if the folder is correct.")
    exit()

# Loop through each category (e.g., Abuse, Arrest, Arson, etc.)
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    # Ensure it's a directory
    if os.path.isdir(category_path):
        print(f"\n📂 Category: {category}")

        # Loop through each video folder inside the category
        for video_folder in os.listdir(category_path):
            video_folder_path = os.path.join(category_path, video_folder)

            # Check if it's a folder (not a file)
            if os.path.isdir(video_folder_path):
                print(f"  📁 Video Folder: {video_folder}")

                # Loop through actual video files inside the subfolder
                for video_file in os.listdir(video_folder_path):
                    video_file_path = os.path.join(video_folder_path, video_file)

                    # Print the video file found
                    print(f"     🎥 Video: {video_file}")


# %%
import os

# Define dataset path
dataset_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset"

# Output file to store the dataset structure
output_file = "dataset_structure.txt"

# Check if path exists
if not os.path.exists(dataset_path):
    print("❌ Path does not exist. Check if the folder is correct.")
    exit()

# Open a file to save the dataset structure
with open(output_file, "w", encoding="utf-8") as file:
    for category in sorted(os.listdir(dataset_path)):  # Sort categories alphabetically
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            print(f"\n📂 Category: {category}")
            file.write(f"\n📂 Category: {category}\n")

            # Loop through each video folder inside the category
            for video_folder in sorted(os.listdir(category_path)):  # Sort video folders
                video_folder_path = os.path.join(category_path, video_folder)
                if os.path.isdir(video_folder_path):
                    print(f"  📁 Video Folder: {video_folder}")
                    file.write(f"  📁 Video Folder: {video_folder}\n")

                    # Loop through video files inside the subfolder
                    for video_file in sorted(os.listdir(video_folder_path)):  # Sort files
                        print(f"     🎥 Video: {video_file}")
                        file.write(f"     🎥 Video: {video_file}\n")

print(f"\n✅ Dataset structure saved to {output_file}")


# %%
import os
import cv2

# Define paths
dataset_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset"
output_path = r"C:\Users\iftek\OneDrive\Desktop\mojor project\Frames"  # Save frames here

# Frame extraction settings
frame_interval = 10  # Extract every 10th frame

# Check if dataset path exists
if not os.path.exists(dataset_path):
    print("❌ Dataset path does not exist. Check your folder.")
    exit()

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Loop through each category (Abuse, Arrest, etc.)
for category in sorted(os.listdir(dataset_path)):
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):
        continue  # Skip if not a folder

    print(f"\n📂 Processing category: {category}")

    # Create category folder in Frames directory
    category_output_path = os.path.join(output_path, category)
    os.makedirs(category_output_path, exist_ok=True)

    # Loop through each video folder
    for video_folder in sorted(os.listdir(category_path)):
        video_folder_path = os.path.join(category_path, video_folder)
        if not os.path.isdir(video_folder_path):
            continue  # Skip if not a folder

        print(f"  📁 Processing video folder: {video_folder}")

        # Create video folder in Frames directory
        video_output_path = os.path.join(category_output_path, video_folder)
        os.makedirs(video_output_path, exist_ok=True)

        # Loop through each video file inside the folder
        for video_file in sorted(os.listdir(video_folder_path)):
            video_file_path = os.path.join(video_folder_path, video_file)

            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue  # Skip non-video files

            print(f"     🎥 Extracting frames from: {video_file}")

            # Open video file
            cap = cv2.VideoCapture(video_file_path)
            frame_count = 0
            saved_frame_count = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break  # Stop if no more frames

                if frame_count % frame_interval == 0:  # Save every Nth frame
                    frame_filename = f"frame_{saved_frame_count:04d}.jpg"
                    frame_path = os.path.join(video_output_path, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_frame_count += 1

                frame_count += 1

            cap.release()  # Release video file

print("\n✅ Frame extraction complete! Frames saved in:", output_path)


# %%
import tensorflow as tf
print(tf.__version__)  # Should print the TensorFlow version


# %%
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# Path to extracted frames
frames_dir = r"C:\Users\iftek\OneDrive\Desktop\mojor project\frames"

# Load the pre-trained ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Resize frames to match ResNet50 input (224x224)
IMG_SIZE = (224, 224)

# Dictionary to store extracted features
features_dict = {}

# Loop through categories (Abuse, Assault, etc.)
for category in os.listdir(frames_dir):
    category_path = os.path.join(frames_dir, category)
    
    if os.path.isdir(category_path):  # Ensure it's a folder
        print(f"🔍 Processing category: {category}")
        features_dict[category] = []
        
        # Loop through frame images
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = preprocess_input(np.expand_dims(img, axis=0))  # Normalize for ResNet50

            # Extract features
            features = model.predict(img)
            features_dict[category].append(features.flatten())  # Flatten to 1D vector

# Save extracted features as a NumPy file
np.save("features.npy", features_dict)
print("✅ Feature extraction complete! Features saved as 'features.npy'.")


# %%
import pandas as pd
import os

# Path to the Labels folder
labels_dir = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset\Labels"

# Dictionary to store label data
labels_data = {}

# Loop through all CSV files in the Labels directory
for file in os.listdir(labels_dir):
    if file.endswith(".csv"):  # Ensure it's a CSV file
        activity_name = os.path.splitext(file)[0]  # Remove .csv to get activity name
        file_path = os.path.join(labels_dir, file)  # Full file path
        
        # Read the CSV file
        labels_data[activity_name] = pd.read_csv(file_path)

# Display the keys (activity names) and preview one dataset
print("Loaded activity labels:", labels_data.keys())
print(labels_data[list(labels_data.keys())[0]].head())  # Show first activity's labels


# %%
import numpy as np

# Load the extracted features
features = np.load("features.npy", allow_pickle=True).item()

# Dictionary to store mapped features and labels
X = []  # Features
y = []  # Labels

# Loop through each activity's label data
for activity, df in labels_data.items():
    for _, row in df.iterrows():
        video_name = row.iloc[0]  # First column contains video name
        label = row.iloc[-1]      # Last column contains label (0 or 1)
        
        # Check if the video has extracted features
        if video_name in features:
            X.append(features[video_name])
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save the processed dataset
np.save("X.npy", X)
np.save("y.npy", y)

print(f"✅ Mapped {len(X)} videos to their labels and saved as 'X.npy' and 'y.npy'.")


# %%
import numpy as np
features = np.load("features.npy", allow_pickle=True).item()
print("Extracted Features Keys:", list(features.keys())[:10])  # Print first 10 keys
print("Total Extracted Features:", len(features))


# %%
from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' = nano, can also use yolov8s.pt, yolov8m.pt

# Open webcam (or video file)
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 inference
    results = model(frame)

    # Draw detection boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            
            # Only show detections with high confidence (e.g., >0.5)
            if conf > 0.5:
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# %%
import os
import shutil
import random

# Paths
dataset_folder = "DCSASS_Dataset/frames"
train_folder = "DCSASS_Dataset/train/images"
val_folder = "DCSASS_Dataset/val/images"
test_folder = "DCSASS_Dataset/test/images"

# Create folders
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all images
all_images = []
for subdir, _, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith(".jpg"):
            all_images.append(os.path.join(subdir, file))

# Shuffle dataset
random.shuffle(all_images)

# Split
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
train_split = int(len(all_images) * train_ratio)
val_split = train_split + int(len(all_images) * val_ratio)

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]

# Move images
for img_path in train_images:
    shutil.move(img_path, os.path.join(train_folder, os.path.basename(img_path)))

for img_path in val_images:
    shutil.move(img_path, os.path.join(val_folder, os.path.basename(img_path)))

for img_path in test_images:
    shutil.move(img_path, os.path.join(test_folder, os.path.basename(img_path)))

print("✅ Dataset split completed!")


# %%
import os
import pandas as pd

# Define paths
dataset_folder = "C:/Users/iftek/OneDrive/Desktop/mojor project/DCSASS Dataset"
labels_folder = os.path.join(dataset_folder, "Labels")
output_labels_folder = os.path.join(dataset_folder, "YOLO_Labels")

# Create output labels folder if not exists
os.makedirs(output_labels_folder, exist_ok=True)

# Get all CSV files in the Labels folder
csv_files = [f for f in os.listdir(labels_folder) if f.endswith(".csv")]

# Assign class IDs dynamically
activity_classes = {activity.replace(".csv", ""): idx for idx, activity in enumerate(csv_files)}
print("Class Mapping:", activity_classes)

# Process each CSV file
for csv_file in csv_files:
    csv_path = os.path.join(labels_folder, csv_file)
    activity_name = csv_file.replace(".csv", "")
    class_id = activity_classes[activity_name]  # Assign class ID dynamically

    # Load CSV file
    df = pd.read_csv(csv_path, header=None, names=["filename", "class", "label"])

    # Convert to YOLO format
    for _, row in df.iterrows():
        image_name = row["filename"]
        label = row["label"]  # 0 or 1

        # Save as a TXT file with the same name as the image
        label_file = os.path.join(output_labels_folder, f"{image_name}.txt")
        with open(label_file, "w") as f:
            f.write(f"{class_id} {label}\n")  # YOLO format: class_id label

print("✅ All activities converted to YOLO format in:", output_labels_folder)


# %%
import os
import yaml

# Define paths
dataset_folder = "C:/Users/iftek/OneDrive/Desktop/mojor project/DCSASS Dataset"
train_folder = os.path.join(dataset_folder, "train/images")
val_folder = os.path.join(dataset_folder, "val/images")
test_folder = os.path.join(dataset_folder, "test/images")

# Get activity names from CSV files
labels_folder = os.path.join(dataset_folder, "Labels")
csv_files = [f.replace(".csv", "") for f in os.listdir(labels_folder) if f.endswith(".csv")]

# Create YAML data
data = {
    "train": train_folder,
    "val": val_folder,
    "test": test_folder,
    "nc": len(csv_files),  # Number of classes
    "names": csv_files,    # Class names
}

# Save to data.yaml
yaml_path = os.path.join(dataset_folder, "data.yaml")
with open(yaml_path, "w") as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"✅ data.yaml created at {yaml_path}")


# %%
import os

# Define dataset directories
dataset_folder = "C:/Users/iftek/OneDrive/Desktop/mojor project/DCSASS_Dataset"
folders = ["train/images", "val/images", "test/images"]

# Check for images
for folder in folders:
    path = os.path.join(dataset_folder, folder)
    
    if not os.path.exists(path):
        print(f"❌ Folder not found: {path}")
        continue

    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if image_files:
        print(f"✅ {folder} contains {len(image_files)} images.")
    else:
        print(f"⚠️ {folder} is empty. No images found!")


# %%
import os
import shutil

# Paths to your dataset
frames_root = r"C:\Users\iftek\OneDrive\Desktop\mojor project\Frames"
dataset_root = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS_Dataset"

# Define where images should go
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(dataset_root, split, "images"), exist_ok=True)

# Move images
for activity in os.listdir(frames_root):  # Abuse, Fighting, etc.
    activity_path = os.path.join(frames_root, activity)
    
    if os.path.isdir(activity_path):
        for video in os.listdir(activity_path):  # Abuse001_x264.mp4, Abuse002_x264.mp4, etc.
            video_path = os.path.join(activity_path, video)
            
            if os.path.isdir(video_path):  # Check if it's a folder containing frames
                frames = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))]
                frames.sort()  # Ensure they are in order
                
                # Split frames into train/val/test (80% train, 10% val, 10% test)
                total = len(frames)
                train_split = int(0.8 * total)
                val_split = int(0.9 * total)
                
                for i, frame in enumerate(frames):
                    src = os.path.join(video_path, frame)
                    
                    if i < train_split:
                        dst = os.path.join(dataset_root, "train", "images", f"{activity}_{video}_{frame}")
                    elif i < val_split:
                        dst = os.path.join(dataset_root, "val", "images", f"{activity}_{video}_{frame}")
                    else:
                        dst = os.path.join(dataset_root, "test", "images", f"{activity}_{video}_{frame}")
                    
                    shutil.copy(src, dst)

print("✅ Frames moved to train/val/test successfully!")


# %%
import os

labels_root = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset\Labels"

if not os.path.exists(labels_root):
    print(f"❌ Folder not found: {labels_root}")
else:
    print(f"✅ Folder exists: {labels_root}")
    print("Files:", os.listdir(labels_root))  # List all files in the folder


# %%
import os

# Define dataset paths
dataset_root = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS_Dataset"
splits = ["train", "val", "test"]

for split in splits:
    image_folder = os.path.join(dataset_root, split, "images")
    label_folder = os.path.join(dataset_root, split, "labels")

    if not os.path.exists(label_folder):
        print(f"❌ Labels folder missing for {split}")
        continue

    missing_labels = []
    
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg"):
            label_file = image_file.replace(".jpg", ".txt")
            label_path = os.path.join(label_folder, label_file)

            if not os.path.exists(label_path):
                missing_labels.append(label_file)

    print(f"✅ Checked {split}. Missing labels: {len(missing_labels)}")
    if missing_labels:
        print("⚠️ Missing label files:", missing_labels[:10])  # Show first 10 missing files


# %%
import os
import shutil

# Paths
labels_root = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS Dataset\YOLO_Labels"
dataset_root = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS_Dataset"

# Move each label file to its corresponding dataset split
for split in ['train', 'val', 'test']:
    src_labels_folder = labels_root  # Current location of YOLO labels
    dest_labels_folder = os.path.join(dataset_root, split, 'labels')  # Where labels should be
    
    # Ensure the destination folder exists
    os.makedirs(dest_labels_folder, exist_ok=True)
    
    # Move label files
    for file in os.listdir(src_labels_folder):
        if file.endswith('.txt'):  # Ensure it's a label file
            src_path = os.path.join(src_labels_folder, file)
            dest_path = os.path.join(dest_labels_folder, file)

            shutil.move(src_path, dest_path)
            print(f"✅ Moved: {file} → {dest_labels_folder}")


# %%
import os

for split in ['train', 'val', 'test']:
    path = f"C:/Users/iftek/OneDrive/Desktop/mojor project/DCSASS_Dataset/{split}/labels"
    print(f"📂 Checking {split} labels...")
    print(os.listdir(path)[:10])  # Show first 10 label files


# %%
import os
import shutil
import random

# Paths
labels_dir = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS_Dataset\train\labels"
val_dir = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS_Dataset\val\labels"
test_dir = r"C:\Users\iftek\OneDrive\Desktop\mojor project\DCSASS_Dataset\test\labels"

# Ensure directories exist
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all label files
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

# Shuffle to randomize distribution
random.shuffle(label_files)

# Define split ratios
val_ratio = 0.2  # 20% for validation
test_ratio = 0.2  # 20% for testing
val_count = int(len(label_files) * val_ratio)
test_count = int(len(label_files) * test_ratio)

# Move files
val_files = label_files[:val_count]
test_files = label_files[val_count:val_count + test_count]

for file in val_files:
    shutil.move(os.path.join(labels_dir, file), os.path.join(val_dir, file))

for file in test_files:
    shutil.move(os.path.join(labels_dir, file), os.path.join(test_dir, file))

print(f"✅ Moved {len(val_files)} labels to validation set.")
print(f"✅ Moved {len(test_files)} labels to test set.")


# %%
import os

for split in ['train', 'val', 'test']:
    path = f"C:/Users/iftek/OneDrive/Desktop/mojor project/DCSASS_Dataset/{split}/labels"
    print(f"📂 Checking {split} labels...")
    print(os.listdir(path)[:10])  # Show first 10 label files


# %%
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="C:/Users/iftek/OneDrive/Desktop/mojor project/DCSASS Dataset/data.yaml",
            epochs=50,
            batch=8,
            imgsz=640)



