import os
import shutil
import random
import soundfile as sf

# 1. Configuration and Paths
BASE_DIR = "/home/kaloori.shiva/Thesis/TIGER/openSLR_hin_dataset/freesounds/Indic2mix"
TARGET_DIR = os.path.join(BASE_DIR, "mix_language")
COMMON_PATH = "kathbath_freesounds_2mix_output/Libri2Mix/wav16k/min"

LANGUAGES = ["Gujarati", "Telugu", "Hindi", "Tamil", "Punjabi"]

# 2. Target Durations in Seconds (1 Hour = 3600 Seconds)
TARGET_DURATIONS = {
    "tr": 6 * 3600,         # 6 hours per language -> 30 hours total
    "tt": 0.8 * 3600,       # 4/5 hours (48 mins) per lang -> 4 hours total
    "cv": 0.8 * 3600        # 4/5 hours (48 mins) per lang -> 4 hours total
}

def main():
    print("Starting dataset mixing process...\n")

    for lang in LANGUAGES:
        print(f"--- Processing Language: {lang} ---")
        
        for split, target_sec in TARGET_DURATIONS.items():
            # Define Source Paths
            src_split_dir = os.path.join(BASE_DIR, lang, COMMON_PATH, split)
            src_mix = os.path.join(src_split_dir, "mix_both")
            src_s1 = os.path.join(src_split_dir, "s1")
            src_s2 = os.path.join(src_split_dir, "s2")

            # Check if source exists to avoid errors
            if not os.path.exists(src_mix):
                print(f"WARNING: Source path not found -> {src_mix}")
                continue

            # Define Target Paths
            tgt_split_dir = os.path.join(TARGET_DIR, COMMON_PATH, split)
            tgt_mix = os.path.join(tgt_split_dir, "mix_both")
            tgt_s1 = os.path.join(tgt_split_dir, "s1")
            tgt_s2 = os.path.join(tgt_split_dir, "s2")

            # Create target directories if they don't exist
            os.makedirs(tgt_mix, exist_ok=True)
            os.makedirs(tgt_s1, exist_ok=True)
            os.makedirs(tgt_s2, exist_ok=True)

            # Get all .wav files from mix_both
            wav_files = [f for f in os.listdir(src_mix) if f.endswith('.wav')]
            
            # Shuffle files to ensure a random selection of audio
            random.shuffle(wav_files)

            current_duration = 0.0
            files_copied = 0

            for wav in wav_files:
                # Stop if we hit the target duration
                if current_duration >= target_sec:
                    break

                wav_path = os.path.join(src_mix, wav)
                
                # Get duration of the audio file in seconds
                try:
                    duration = sf.info(wav_path).duration
                except Exception as e:
                    print(f"Error reading {wav_path}: {e}")
                    continue
                
                current_duration += duration
                files_copied += 1

                # Copy the exact same file for mix_both, s1, and s2
                shutil.copy2(os.path.join(src_mix, wav), os.path.join(tgt_mix, wav))
                shutil.copy2(os.path.join(src_s1, wav), os.path.join(tgt_s1, wav))
                shutil.copy2(os.path.join(src_s2, wav), os.path.join(tgt_s2, wav))

            # Print status update
            actual_hours = current_duration / 3600
            print(f"[{split}] Copied {files_copied} files | Total duration: {actual_hours:.2f} hours")
        print("\n")
        
    print("✅ Dataset creation completed successfully!")

if __name__ == "__main__":
    main()