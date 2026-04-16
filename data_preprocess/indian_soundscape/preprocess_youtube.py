import os
import random
import warnings
import torch
import librosa
import soundfile as sf
import pyloudnorm as pdn
import time
import glob
import yt_dlp

warnings.filterwarnings("ignore")

# --- 1. GLOBAL CONFIGURATION ---
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=O63Y54h1D7g&list=PLMx-Lf2LPYsvXTXl2MBitGkBkU4089B_Q",
    "https://youtube.com/playlist?list=PLjYJhgD_UErciagTpFhe91uAj7Dc9f1KH",
    "https://youtube.com/playlist?list=PLMx-Lf2LPYsvQNRabiTl9wmJzuF73au0r",
    "https://www.youtube.com/watch?v=eX3ioEkKpH0",
    "https://youtube.com/playlist?list=PLvFFO2bJlAuEQuk9aWppz9JF-7PXosdUI",
    "https://www.youtube.com/watch?v=pqmIl6tdOds&list=PLvFFO2bJlAuFHSQfiu82sAh2aUw0kc3Qa",
    "https://www.youtube.com/playlist?list=PLvFFO2bJlAuGnEWume5GkzksG7Xv7Gdqq",
    "https://youtube.com/playlist?list=PLvFFO2bJlAuFu7ie96AiEc0cG6S32MMso",
    "https://youtu.be/BO533LAp9HQ",
    "https://youtube.com/playlist?list=PL2dFrbAAmDPE63mGSxsb5-tiqPIXGj5A8"
]

INPUT_DIR = "india_soundscapes_raw_new"
BASE_OUTPUT_DIR = "india_noise_youtube"

# TARGET_SR must stay at 16000 for Silero VAD to function correctly
TARGET_SR = 16000  
TARGET_LUFS = -23.0 
SPEECH_THRESHOLD = 0.05  
CHUNK_SEC = 12
OVERLAP_SEC = 4.0 # Changed to 4 seconds

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
for s in ['tr', 'cv', 'tt']: 
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, s), exist_ok=True)

# --- 2. VAD SETUP ---
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, _, _, _) = utils

# --- 3. PROCESSING LOGIC ---
def process_and_save(file_path, split_name, meter):
    """Chunks audio, normalizes LUFS, filters via VAD, and saves."""
    chunks_saved = 0
    try:
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        input_lufs = meter.integrated_loudness(audio)
        if input_lufs < -70: return 0 
        audio_norm = pdn.normalize.loudness(audio, input_lufs, TARGET_LUFS)
        
        chunk_samples = CHUNK_SEC * TARGET_SR
        hop_samples = int((CHUNK_SEC - OVERLAP_SEC) * TARGET_SR) # 12 - 4 = 8 sec hop
        
        if len(audio_norm) < chunk_samples: return 0
        
        frames = librosa.util.frame(audio_norm, frame_length=chunk_samples, hop_length=hop_samples)
        
        for i in range(frames.shape[1]):
            chunk = frames[:, i]
            chunk_tensor = torch.from_numpy(chunk).float()
            
            speech_timestamps = get_speech_timestamps(chunk_tensor, vad_model, sampling_rate=TARGET_SR)
            speech_duration = sum([t['end'] - t['start'] for t in speech_timestamps])
            
            if (speech_duration / chunk_samples) <= SPEECH_THRESHOLD:
                base_name = os.path.basename(file_path).replace(' ', '_').rsplit('.', 1)[0]
                out_name = f"{base_name}_ch{i}.wav"
                out_path = os.path.join(BASE_OUTPUT_DIR, split_name, out_name)
                sf.write(out_path, chunk, TARGET_SR, subtype='PCM_16')
                chunks_saved += 1
                
    except Exception as e:
        print(f"  [ERROR] Processing {file_path}: {e}")
    return chunks_saved

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    meter = pdn.Meter(TARGET_SR)
    
    print("Extracting individual video URLs from playlists...")
    all_video_urls = []
    
    # Extract flat lists to avoid downloading everything at once
    ydl_extract_opts = {'extract_flat': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_extract_opts) as ydl:
        for url in YOUTUBE_URLS:
            try:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:
                    for entry in info['entries']:
                        if entry:
                            vid_url = entry.get('url')
                            if not vid_url and entry.get('id'):
                                vid_url = f"https://www.youtube.com/watch?v={entry['id']}"
                            if vid_url:
                                all_video_urls.append(vid_url)
                else:
                    all_video_urls.append(info['original_url'] if 'original_url' in info else url)
            except Exception as e:
                print(f"Failed to extract info for {url}: {e}")

    # Remove duplicates
    all_video_urls = list(set(all_video_urls))
    print(f"Found {len(all_video_urls)} unique YouTube videos.")

    # Randomize to prevent data leakage across splits
    random.seed(42)
    random.shuffle(all_video_urls)
    
    # Split 15% TT, 15% CV, 70% TR (Video-level split!)
    num_sounds = len(all_video_urls)
    tt_end = int(0.15 * num_sounds)
    cv_end = tt_end + int(0.15 * num_sounds)
    
    splits = [
        ('tt', all_video_urls[:tt_end]), 
        ('cv', all_video_urls[tt_end:cv_end]), 
        ('tr', all_video_urls[cv_end:])
    ]

    # yt-dlp download options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(INPUT_DIR, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'ignoreerrors': True
    }

    total_chunks_saved = 0

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for split_name, video_list in splits:
            print(f"\n--- Processing {split_name.upper()} split ({len(video_list)} source videos) ---")
            
            for vid_url in video_list:
                print(f"[{split_name}] Downloading & Processing: {vid_url}")
                try:
                    # Download the video audio as .wav
                    ydl.download([vid_url])
                    
                    # Find the downloaded file in INPUT_DIR
                    downloaded_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
                    
                    for f in downloaded_files:
                        saved = process_and_save(f, split_name, meter)
                        print(f"  -> Generated {saved} valid chunks.")
                        total_chunks_saved += saved
                        
                        # CRITICAL: Delete raw file immediately to save server space
                        os.remove(f) 
                        
                except Exception as e:
                    print(f"  [SKIP] Failed on {vid_url}: {e}")

    print(f"\nProcessing Complete! Total 12s chunks generated: {total_chunks_saved}")