import os
import librosa
import soundfile as sf
import numpy as np
import random
import shutil

# 1. Configuration
INPUT_DIR = "india_soundscapes_dataset"
BASE_OUTPUT_DIR = "india_noise_new"
CHUNK_SEC = 5
OVERLAP_SEC = 3
TARGET_SR = 16000

# WHAM! Typical Split Ratios
SPLITS = {'tr': 0.80, 'cv': 0.1, 'tt': 0.1}

# 2. Prepare Directory Structure
for s in SPLITS.keys():
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, s), exist_ok=True)

# 3. Get and Shuffle Source Files
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp3")]
random.seed(42) # For reproducibility
random.shuffle(all_files)

# 4. Partition the Files
num_files = len(all_files)
tr_end = int(SPLITS['tr'] * num_files)
cv_end = tr_end + int(SPLITS['cv'] * num_files)

file_groups = {
    'tr': all_files[:tr_end],
    'cv': all_files[tr_end:cv_end],
    'tt': all_files[cv_end:]
}

# 5. Process and Chunk
hop_samples = (CHUNK_SEC - OVERLAP_SEC) * TARGET_SR
chunk_samples = CHUNK_SEC * TARGET_SR

for split_name, files in file_groups.items():
    print(f"\n--- Processing {split_name} set ({len(files)} files) ---")
    split_dir = os.path.join(BASE_OUTPUT_DIR, split_name)
    
    for file_name in files:
        path = os.path.join(INPUT_DIR, file_name)
        try:
            audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            
            if len(audio) > chunk_samples:
                frames = librosa.util.frame(audio, frame_length=chunk_samples, hop_length=hop_samples)
                
                for i in range(frames.shape[1]):
                    chunk = librosa.util.normalize(frames[:, i])
                    chunk_id = file_name.replace(".mp3", "")
                    output_name = f"{chunk_id}_chunk{i}.wav"
                    sf.write(os.path.join(split_dir, output_name), chunk, TARGET_SR)
        except Exception as e:
            print(f"Error on {file_name}: {e}")

print(f"\nDone! Dataset created in {BASE_OUTPUT_DIR}")