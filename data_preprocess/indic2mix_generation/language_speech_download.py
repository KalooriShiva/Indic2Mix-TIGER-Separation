# ==============================================================================
# Copyright (c) 2026 Kaloori Shiva Prasad
#
# Script: language_speech_download.py
# Description: Automates the downloading and extraction of the native Indic 
#              Kathbath corpus via Hugging Face Datasets. Dynamically groups
#              and splits speakers based on target file distributions. 
#              Subsequently applies the pretrained Demucs (dns64) time-domain 
#              audio separation model to denoise the raw speech files, ensuring 
#              pristine single-speaker signals prior to Indic2Mix dataset mixing.
# ==============================================================================

import os
import torch
import soundfile as sf
from datasets import load_dataset, Audio
from collections import defaultdict
from tqdm import tqdm
from denoiser import pretrained
from denoiser.dsp import convert_audio

# ================= CONFIG =================

LANGUAGE = "tamil"

OUTPUT_BASE = "Indic2Mix"

TARGET_SIZE = 8000
TARGET_SPEAKERS = 10

# MAX_FILES = 50
# TARGET_SIZE = 100
# TARGET_SPEAKERS = 2

# ==========================================


def build_dataset():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Running on: {device}")

    # -------------------------
    # Load denoiser model
    # -------------------------

    print("📥 Loading Demucs model...")

    model = pretrained.dns64().to(device)
    model.eval()

    # =========================
    # PHASE 1 — Count Speakers
    # =========================

    print(
        f"\n🚀 PHASE 1: Counting speakers ({LANGUAGE})..."
    )

    ds = load_dataset(
        "ai4bharat/Kathbath",
        LANGUAGE,
        streaming=True
    )

    speaker_counts = defaultdict(int)

    for split in ds.keys():

        for sample in tqdm(
            ds[split],
            desc=f"Counting {split}",
            unit=" files"
        ):

            spk_id = sample.get("speaker_id")

            if spk_id:

                speaker_counts[spk_id] += 1

    print(
        f"✅ Found {len(speaker_counts)} speakers"
    )

    # =========================
    # PHASE 2 — Assign Splits
    # =========================

    print(
        "\n🧠 PHASE 2: Assigning speakers..."
    )

    ideal_size = TARGET_SIZE // TARGET_SPEAKERS

    sorted_speakers = sorted(
        speaker_counts.items(),
        key=lambda x: abs(
            x[1] - ideal_size
        )
    )

    speaker_mapping = {}

    head_counts = {
        "test": 0,
        "valid": 0,
        "train": 0
    }

    file_counts = {
        "test": 0,
        "valid": 0,
        "train": 0
    }

    for spk_id, count in sorted_speakers:

        if head_counts["test"] < TARGET_SPEAKERS:

            speaker_mapping[spk_id] = "test"
            head_counts["test"] += 1
            file_counts["test"] += count

        elif head_counts["valid"] < TARGET_SPEAKERS:

            speaker_mapping[spk_id] = "valid"
            head_counts["valid"] += 1
            file_counts["valid"] += count

        else:

            speaker_mapping[spk_id] = "train"
            head_counts["train"] += 1
            file_counts["train"] += count

    print("\nSplit summary:")

    print(
        f"🧪 TEST  : {file_counts['test']} files "
        f"({head_counts['test']} speakers)"
    )

    print(
        f"📐 VALID : {file_counts['valid']} files "
        f"({head_counts['valid']} speakers)"
    )

    print(
        f"🏋️ TRAIN : {file_counts['train']} files "
        f"({head_counts['train']} speakers)"
    )

    # =========================
    # PHASE 3–5 Processing
    # =========================

    print(
        "\n🎧 PHASE 3–5: Download + Denoise + Save..."
    )

    lang_out_dir = os.path.join(
        OUTPUT_BASE,
        LANGUAGE.capitalize()
    )

    for split in ["train", "valid", "test"]:

        os.makedirs(
            os.path.join(
                lang_out_dir,
                split
            ),
            exist_ok=True
        )

    saved_count = 0
    skipped_count = 0

    ds_audio = load_dataset(
        "ai4bharat/Kathbath",
        LANGUAGE,
        streaming=True
    )

    # 🔧 Important — enable decoding
    ds_audio["train"] = ds_audio["train"].cast_column(
        "audio_filepath",
        Audio()
    )

    ds_audio["valid"] = ds_audio["valid"].cast_column(
        "audio_filepath",
        Audio()
    )

    for split in ds_audio.keys():

        for sample in tqdm(
            ds_audio[split],
            desc=f"Processing {split}",
            unit=" files"
        ):

            spk_id = sample.get("speaker_id")

            if spk_id not in speaker_mapping:
                continue

            assigned_split = speaker_mapping[spk_id]

            try:

                # -------------------------
                # PHASE 3 — Download
                # -------------------------

                audio = sample["audio_filepath"]

                audio_array = audio["array"]
                sr = audio["sampling_rate"]

                wav = torch.tensor(
                    audio_array
                ).float().to(device)

                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)

                # -------------------------
                # PHASE 4 — Denoise
                # -------------------------

                wav = convert_audio(
                    wav,
                    sr,
                    model.sample_rate,
                    model.chin
                )

                with torch.no_grad():

                    clean = model(
                        wav.unsqueeze(0)
                    ).squeeze(0)

                # Force mono
                if clean.shape[0] > 1:

                    clean = clean.mean(
                        dim=0,
                        keepdim=True
                    )

                # -------------------------
                # PHASE 5 — Save
                # -------------------------

                fname = sample.get(
                    "fname",
                    f"audio_{saved_count}.wav"
                )

                filename = (
                    os.path.splitext(fname)[0]
                    + ".wav"
                )

                speaker_dir = os.path.join(
                    lang_out_dir,
                    assigned_split,
                    str(spk_id)
                )

                os.makedirs(
                    speaker_dir,
                    exist_ok=True
                )

                output_path = os.path.join(
                    speaker_dir,
                    filename
                )

                sf.write(
                    output_path,
                    clean.cpu().numpy().T,
                    model.sample_rate,
                    subtype="PCM_16"
                )

                saved_count += 1

                # Optional test stop
                if (
                    MAX_FILES
                    and saved_count >= MAX_FILES
                ):

                    print(
                        "\n🛑 Test limit reached."
                    )

                    return

            except Exception:

                skipped_count += 1

    print(
        f"\n✅ SUCCESS! {saved_count} files saved."
    )

    print(
        f"⚠️ Skipped {skipped_count} files."
    )


if __name__ == "__main__":

    build_dataset()
