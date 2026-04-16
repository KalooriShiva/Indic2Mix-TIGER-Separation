# ==============================================================================
# Copyright (c) 2020 Joris Cosentino et al. (Original LibriMix Implementation)
# 
# This script has been modified by Kaloori Shiva Prasad for the Indic2Mix dataset.
# Modifications include: 
# - Adapting directory structures for the Kathbath corpus.
# - Implementing custom LUFS offsets for the Indian Soundscape noise.
# - Modifying the pair generation logic for monolingual Indic mixtures.
# ==============================================================================
import os
import argparse
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm
import librosa
# Global parameter
NUMBER_OF_SECONDS = 3
RATE = 16000  # sampling rate

# Speaker gender dictionary
SPEAKER_GENDER = {
    878: "F",
    154: "F",
    870: "F",
    90: "F",
    807: "F",
    67: "F",
    507: "F",
    837: "F",
    764: "F",
    781: "F",
    21: "M",
    797: "M",
    302: "M",
    116: "M",
    510: "M",
    491: "M",
    655: "M",
    84: "M",
    739: "M",
    401: "M",
    203: "F",
    1099: "F",
    347: "F",
    1042: "F",
    1029: "F",
    80: "F",
    793: "F",
    925: "F",
    307: "F",
    872: "F",
    129: "F",
    106: "F",
    1196: "F",
    750: "F",
    1146: "F",
    133: "F",
    827: "F",
    412: "F",
    900: "F",
    1142: "F",
    146: "F",
    794: "F",
    565: "M",
    674: "M",
    1162: "M",
    941: "M",
    697: "M",
    198: "M",
    214: "M",
    712: "M",
    287: "M",
    470: "M",
    1197: "M",
    690: "M",
    246: "M",
    632: "M",
    880: "M",
    603: "M",
    49: "M",
    218: "M",
    864: "M",
    1047: "M",
    495: "M",
    936: "M",
    1157: "M",
    951: "M",
    453: "M",
    271: "M",
    1091: "M",
    489: "M",
    125: "M",
    138: "M",
    8: "M",
    1184: "M",
    285: "M",
    340: "M",
    486: "M",
    1107: "M",
    675: "M",
    1071: "M",
    1098: "M",
    353: "M",
    1167: "M",
    483: "M",
    50: "M",
    707: "M",
    473: "M",
    46: "M",
    957: "M",
    424: "M",
    499: "M",
    283: "M",
    437: "M",
    1100: "M",
    207: "M",
    995: "M",
    73: "M",
    714: "M",
    1028: "M",
    392: "M",
    868: "M",
    686: "M",
    634: "M",
    1089: "M",
    983: "M",
    1072: "M",
    1080: "M",
    137: "M",
    494: "M",
    244: "M",
    669: "M",
    826: "M",
    350: "M",
    319: "M",
    1193: "M",
    398: "M",
    466: "M",
    585: "M",
    232: "M",
    343: "M",
    588: "M",
    928: "M",
    199: "M",
    173: "M",
    461: "M",
    391: "M",
    327: "M",
    698: "M",
    118: "M",
    10: "M",
    434: "M",
    77: "M",
    1086: "M",
    435: "M",
    165: "M",
    280: "M",
    974: "M",
    409: "M"
}
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--kathbath_dir', type=str, required=True,
                    help='Path to Kathbath root directory')


def main(args):
    kathbath_dir = args.kathbath_dir
    md_dir = os.path.join(kathbath_dir, 'metadata')
    os.makedirs(md_dir, exist_ok=True)
    
    subsets = ['tr', 'tt', 'cv']
    for subset in subsets:
        subset_path = os.path.join(kathbath_dir, subset)
        create_kathbath_metadata(subset_path, md_dir, subset)


def create_kathbath_metadata(subset_path, md_dir, subset_name):
    """Generate metadata for a Kathbath subset"""
    print(f"Creating metadata for {subset_name}")
    
    # Recursively look for all audio files
    audio_files = glob.glob(os.path.join(subset_path, '**/*.wav'), recursive=True)
    
    # Prepare dataframe
    df = pd.DataFrame(columns=['speaker_ID', 'sex', 'subset', 'length', 'origin_path'])
    
    for audio_path in tqdm(audio_files, total=len(audio_files)):
        # Extract speaker ID from folder name
        speaker_id = int(os.path.basename(os.path.dirname(audio_path)))
        sex = SPEAKER_GENDER.get(speaker_id, 'Unknown')
        length = len(sf.SoundFile(audio_path))
        rel_path = os.path.relpath(audio_path, os.path.dirname(md_dir))  # relative to kathbath root
        if length >= NUMBER_OF_SECONDS * RATE:
            df.loc[len(df)] = [speaker_id, sex, subset_name, length, rel_path]
    
    # Sort by length
    df = df.sort_values('length')
    
    # Save to CSV
    save_path = os.path.join(md_dir, f"{subset_name}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved metadata to {save_path}")

# def create_kathbath_metadata(subset_path, md_dir, subset_name):
#     """Generate metadata for a Kathbath subset with Silence Filtering"""
#     print(f"Creating metadata for {subset_name}")
    
#     # Recursively look for all audio files
#     audio_files = glob.glob(os.path.join(subset_path, '**/*.wav'), recursive=True)
    
#     # Prepare dataframe
#     df = pd.DataFrame(columns=['speaker_ID', 'sex', 'subset', 'length', 'origin_path'])
    
#     for audio_path in tqdm(audio_files, total=len(audio_files)):
#         # Extract speaker ID from folder name
#         speaker_id = int(os.path.basename(os.path.dirname(audio_path)))
#         sex = SPEAKER_GENDER.get(speaker_id, 'Unknown')
        
#         # --- CHANGE START: Trim silence before calculating length ---
#         try:
#             # Load with librosa to trim silence
#             y, _ = librosa.load(audio_path, sr=RATE)
#             y_trimmed, _ = librosa.effects.trim(y, top_db=30)
#             active_length = len(y_trimmed)
            
#             # Use active_length instead of file length
#             if active_length >= NUMBER_OF_SECONDS * RATE:
#                 rel_path = os.path.relpath(audio_path, os.path.dirname(md_dir))
#                 df.loc[len(df)] = [speaker_id, sex, subset_name, active_length, rel_path]
#         except Exception as e:
#             print(f"Skipping {audio_path}: {e}")
#         # --- CHANGE END ---
    
    # # Sort by length
    # df = df.sort_values('length')
    
    # # Save to CSV
    # save_path = os.path.join(md_dir, f"{subset_name}.csv")
    # df.to_csv(save_path, index=False)
    # print(f"Saved metadata to {save_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
