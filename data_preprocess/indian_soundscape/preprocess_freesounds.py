
import os
import random
import warnings
import torch
import librosa
import soundfile as sf
import pyloudnorm as pdn
import freesound
import time

warnings.filterwarnings("ignore")

# --- 1. GLOBAL CONFIGURATION ---
# Using the direct API Key for server compatibility (No browser needed)
MY_API_KEY = "FREESOUNDS_API_KEY" 

INPUT_DIR = "india_soundscapes_raw_new"
BASE_OUTPUT_DIR = "india_noise_wham_new_12_sec_chunk"
TARGET_SR = 16000  
TARGET_LUFS = -23.0 
SPEECH_THRESHOLD = 0.05  # Ratio of speech allowed in a "noise" chunk
CHUNK_SEC = 12
OVERLAP_SEC = 4.0

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
for s in ['tr', 'cv', 'tt']: 
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, s), exist_ok=True)

# --- 2. VAD SETUP ---
# Loading Silero VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, _, _, _) = utils

# --- 3. AUTHENTICATION (SIMPLIFIED FOR SERVER) ---
def get_authorized_client():
    fs_client = freesound.FreesoundClient()
    # auth_type "token" bypasses the need for OAuth2/browser login
    fs_client.set_token(MY_API_KEY, "token")
    return fs_client

# --- 4. PROCESSING LOGIC ---
def process_and_save(file_path, split_name, meter):
    """Chunks audio, normalizes LUFS, filters via VAD, and saves."""
    chunks_saved = 0
    try:
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # Measure and Normalize Loudness
        input_lufs = meter.integrated_loudness(audio)
        if input_lufs < -70: return 0 # Skip near-silent files
        audio_norm = pdn.normalize.loudness(audio, input_lufs, TARGET_LUFS)
        
        chunk_samples = CHUNK_SEC * TARGET_SR
        hop_samples = int((CHUNK_SEC - OVERLAP_SEC) * TARGET_SR)
        
        if len(audio_norm) < chunk_samples: return 0
        
        # Create sliding window frames
        frames = librosa.util.frame(audio_norm, frame_length=chunk_samples, hop_length=hop_samples)
        
        for i in range(frames.shape[1]):
            chunk = frames[:, i]
            chunk_tensor = torch.from_numpy(chunk).float()
            
            # VAD check: Is there too much speech in this 'noise' chunk?
            speech_timestamps = get_speech_timestamps(chunk_tensor, vad_model, sampling_rate=TARGET_SR)
            speech_duration = sum([t['end'] - t['start'] for t in speech_timestamps])
            
            if (speech_duration / chunk_samples) <= SPEECH_THRESHOLD:
                out_name = f"{os.path.basename(file_path).replace(' ', '_').split('.')[0]}_ch{i}.wav"
                out_path = os.path.join(BASE_OUTPUT_DIR, split_name, out_name)
                sf.write(out_path, chunk, TARGET_SR, subtype='PCM_16')
                chunks_saved += 1
                
    except Exception as e:
        print(f"  [ERROR] Processing {file_path}: {e}")
    return chunks_saved

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    client = get_authorized_client()
    meter = pdn.Meter(TARGET_SR)
    
    print("Searching Freesound for 'India Soundscapes'...")
    all_sounds = []
    current_page = 1
    
    # Collect metadata for all matching sounds
    while True:
        try:
            results = client.search(
                query="india", 
                filter="category:Soundscapes", 
                fields="id,name,download", 
                page_size=100, 
                page=current_page
            )
            all_sounds.extend([s for s in results])
            if not results.next or current_page > 10: # Limit pages to avoid huge API wait
                break
            current_page += 1
        except Exception as e:
            print(f"Search error: {e}")
            break

    print(f"Found {len(all_sounds)} unique sound files.")

    # RANDOMIZE BEFORE SPLITTING (Prevents data leakage/ordering bias)
    random.seed(42)
    random.shuffle(all_sounds)
    
    # Split the original sound sources (15% TT, 15% CV, 70% TR)
    num_sounds = len(all_sounds)
    tt_end = int(0.15 * num_sounds)
    cv_end = tt_end + int(0.15 * num_sounds)
    
    tt_files = all_sounds[:tt_end]
    cv_files = all_sounds[tt_end:cv_end]
    tr_files = all_sounds[cv_end:]

    # Removed the limits from the tuples
    splits = [
        ('tt', tt_files), 
        ('cv', cv_files), 
        ('tr', tr_files)
    ]

    for split_name, sound_list in splits:
        print(f"\n--- Processing {split_name.upper()} split ({len(sound_list)} source files) ---")
        
        for sound in sound_list:
            print(f"[{split_name}] Downloading: {sound.name} (ID: {sound.id})")
            try:
                # Download original HQ file
                sound.retrieve(INPUT_DIR)
                
                # Process all files currently in the input dir
                for f in os.listdir(INPUT_DIR):
                    full_path = os.path.join(INPUT_DIR, f)
                    saved = process_and_save(full_path, split_name, meter)
                    print(f"  -> Generated {saved} valid chunks.")
                    os.remove(full_path) # Clean up raw file immediately
                
                time.sleep(1) # Polite API delay
            except Exception as e:
                print(f"  [SKIP] Download failed: {e}")
                time.sleep(2)

    print("\nProcessing Complete!")
