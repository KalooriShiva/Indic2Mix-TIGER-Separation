# create_kathbathmix_from_metadata.py
import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import functools
from scipy.signal import resample_poly
import tqdm.contrib.concurrent
import hashlib
import re
import librosa

# eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000

parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--wham_dir', type=str, required=True,
                    help='Path to wham_noise root directory')
parser.add_argument('--metadata_dir', type=str, required=True,
                    help='Path to the LibriMix metadata directory')
parser.add_argument('--librimix_outdir', type=str, default=None,
                    help='Path to the desired dataset root directory')
parser.add_argument('--n_src', type=int, required=True,
                    help='Number of sources in mixtures')
parser.add_argument('--freqs', nargs='+', default=['8k', '16k'],
                    help='--freqs 16k 8k will create 2 directories wav8k '
                         'and wav16k')
parser.add_argument('--modes', nargs='+', default=['min', 'max'],
                    help='--modes min max will create 2 directories in '
                         'each freq directory')
parser.add_argument('--types', nargs='+', default=['mix_clean', 'mix_both',
                                                   'mix_single'],
                    help='--types mix_clean mix_both mix_single ')


def main(args):
    librispeech_dir = args.librispeech_dir
    wham_dir = args.wham_dir
    metadata_dir = args.metadata_dir
    librimix_outdir = args.librimix_outdir
    n_src = args.n_src

    if librimix_outdir is None:
        librimix_outdir = os.path.dirname(metadata_dir)
    librimix_outdir = os.path.join(librimix_outdir, f'Libri{n_src}Mix')

    freqs = [freq.lower() for freq in args.freqs]
    modes = [mode.lower() for mode in args.modes]
    types = [t.lower() for t in args.types]

    create_librimix(librispeech_dir, wham_dir, librimix_outdir, metadata_dir,
                    freqs, n_src, modes, types)


def clean_mix_id(mix_id: str) -> str:
    """Sanitize mixture ID before using as filename."""
    s = str(mix_id)
    s = s.replace(".wav", "")            # remove any embedded or trailing .wav
    s = s.replace(os.sep, "_")           # guard against accidental path bits
    s = re.sub(r"\s+", "_", s)           # spaces -> underscore
    s = re.sub(r"_+", "_", s).strip("_") # collapse underscores
    return s


def _safe_filename(basename: str, maxlen: int = 120) -> str:
    """Ensure filename isn't too long for the filesystem."""
    if len(basename) <= maxlen:
        return basename
    h = hashlib.md5(basename.encode()).hexdigest()[:8]
    keep = maxlen // 2
    return f"{basename[:keep]}_{h}"


def _write_wav_safe(abs_path: str, data: np.ndarray, sr: int):
    """Create parent dir, sanitize data, then write."""
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    sf.write(abs_path, data, sr)


def write_sources(mix_id, transformed_sources, subdirs, dir_path, freq, n_src):
    abs_source_path_list = []
    base = clean_mix_id(mix_id)
    ex_filename = _safe_filename(base) + ".wav"
    for src, src_dir in zip(transformed_sources[:n_src], subdirs[:n_src]):
        save_path = os.path.join(dir_path, src_dir, ex_filename)
        abs_save_path = os.path.abspath(save_path)
        _write_wav_safe(abs_save_path, src, freq)
        abs_source_path_list.append(abs_save_path)
    return abs_source_path_list


def write_noise(mix_id, transformed_sources, dir_path, freq):
    base = clean_mix_id(mix_id)
    ex_filename = _safe_filename(base) + ".wav"
    save_path = os.path.join(dir_path, "noise", ex_filename)
    abs_save_path = os.path.abspath(save_path)
    _write_wav_safe(abs_save_path, transformed_sources[-1], freq)
    return abs_save_path


def write_mix(mix_id, mixture, dir_path, subdir, freq):
    base = clean_mix_id(mix_id)
    ex_filename = _safe_filename(base) + ".wav"
    save_path = os.path.join(dir_path, subdir, ex_filename)
    abs_save_path = os.path.abspath(save_path)
    _write_wav_safe(abs_save_path, mixture, freq)
    return abs_save_path


def create_librimix(librispeech_dir, wham_dir, out_dir, metadata_dir,
                    freqs, n_src, modes, types):
    """ Generate sources mixtures and saves them in out_dir"""
    md_filename_list = [f for f in os.listdir(metadata_dir)
                        if f.endswith(".csv") and 'info' not in f]
    for md_filename in md_filename_list:
        csv_path = os.path.join(metadata_dir, md_filename)
        process_metadata_file(csv_path, freqs, n_src, librispeech_dir,
                              wham_dir, out_dir, modes, types)


def process_metadata_file(csv_path, freqs, n_src, librispeech_dir, wham_dir,
                          out_dir, modes, types):
    md_file = pd.read_csv(csv_path, engine='python')

    
    for freq in freqs:
        freq_path = os.path.join(out_dir, 'wav' + freq)
        freq = int(freq.strip('k')) * 1000

        for mode in modes:
            mode_path = os.path.join(freq_path, mode)
            subset_metadata_path = os.path.join(mode_path, 'metadata')
            os.makedirs(subset_metadata_path, exist_ok=True)

            dir_name = os.path.basename(csv_path).replace(
                f'libri{n_src}mix_', '').replace('-clean', '').replace(
                '.csv', '')
            dir_path = os.path.join(mode_path, dir_name)

            if os.path.isdir(dir_path):
                print(f"Directory {dir_path} already exist. Files won't be overwritten")
                continue

            print(f"Creating mixtures and sources from {csv_path} in {dir_path}")

            if types == ['mix_clean']:
                subdirs = [f's{i + 1}' for i in range(n_src)] + ['mix_clean']
            else:
                subdirs = [f's{i + 1}' for i in range(n_src)] + types + ['noise']

            for subdir in subdirs:
                os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)

            process_utterances(md_file, librispeech_dir, wham_dir, freq, mode,
                               subdirs, dir_path, subset_metadata_path, n_src)


def process_utterances(md_file, librispeech_dir, wham_dir, freq, mode, subdirs,
                       dir_path, subset_metadata_path, n_src):
    md_dic = {}
    dir_name = os.path.basename(dir_path)

    for subdir in subdirs:
        if subdir.startswith('mix'):
            md_dic[f'metrics_{dir_name}_{subdir}'] = create_empty_metrics_md(n_src, subdir)
            md_dic[f'mixture_{dir_name}_{subdir}'] = create_empty_mixture_md(n_src, subdir)

    for results in tqdm.contrib.concurrent.process_map(
        functools.partial(
            process_utterance,
            n_src, librispeech_dir, wham_dir, freq, mode, subdirs, dir_path),
        [row for _, row in md_file.iterrows()],
        chunksize=10,
    ):
        for mix_id, snr_list, abs_mix_path, abs_source_path_list, abs_noise_path, length, subdir in results:
            add_to_metrics_metadata(md_dic[f"metrics_{dir_name}_{subdir}"],
                                    mix_id, snr_list)
            add_to_mixture_metadata(md_dic[f'mixture_{dir_name}_{subdir}'],
                                    mix_id, abs_mix_path, abs_source_path_list,
                                    abs_noise_path, length, subdir)

    for md_df in md_dic:
        save_path_mixture = os.path.join(subset_metadata_path, md_df + '.csv')
        md_dic[md_df].to_csv(save_path_mixture, index=False)


def process_utterance(n_src, librispeech_dir, wham_dir, freq, mode, subdirs, dir_path, row):
    res = []
    mix_id, gain_list, sources = read_sources(row, n_src, librispeech_dir, wham_dir)
    transformed_sources = transform_sources(sources, freq, mode, gain_list)

    abs_source_path_list = write_sources(mix_id, transformed_sources, subdirs, dir_path, freq, n_src)
    abs_noise_path = write_noise(mix_id, transformed_sources, dir_path, freq)

    for subdir in subdirs:
        if subdir == 'mix_clean':
            sources_to_mix = transformed_sources[:n_src]
        elif subdir == 'mix_both':
            sources_to_mix = transformed_sources
        elif subdir == 'mix_single':
            sources_to_mix = [transformed_sources[0], transformed_sources[-1]]
        else:
            continue

        mixture = mix(sources_to_mix)
        abs_mix_path = write_mix(mix_id, mixture, dir_path, subdir, freq)
        length = len(mixture)
        snr_list = compute_snr_list(mixture, sources_to_mix)

        res.append((mix_id, snr_list, abs_mix_path, abs_source_path_list,
                    abs_noise_path, length, subdir))

    return res


def create_empty_metrics_md(n_src, subdir):
    metrics_dataframe = pd.DataFrame()
    metrics_dataframe['mixture_ID'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            metrics_dataframe[f"source_{i + 1}_SNR"] = {}
    elif subdir == 'mix_both':
        for i in range(n_src):
            metrics_dataframe[f"source_{i + 1}_SNR"] = {}
        metrics_dataframe[f"noise_SNR"] = {}
    elif subdir == 'mix_single':
        metrics_dataframe["source_1_SNR"] = {}
        metrics_dataframe[f"noise_SNR"] = {}
    return metrics_dataframe


def create_empty_mixture_md(n_src, subdir):
    mixture_dataframe = pd.DataFrame()
    mixture_dataframe['mixture_ID'] = {}
    mixture_dataframe['mixture_path'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_path"] = {}
    elif subdir == 'mix_both':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_path"] = {}
        mixture_dataframe[f"noise_path"] = {}
    elif subdir == 'mix_single':
        mixture_dataframe["source_1_path"] = {}
        mixture_dataframe[f"noise_path"] = {}
    mixture_dataframe['length'] = {}
    return mixture_dataframe


def read_sources(row, n_src, librispeech_dir, wham_dir):
    mixture_id = row['mixture_ID']
    sources_path_list = get_list_from_csv(row, 'source_path', n_src)
    gain_list = get_list_from_csv(row, 'source_gain', n_src)
    sources_list = []
    max_length = 0

    for sources_path in sources_path_list:
        sources_path = os.path.join(librispeech_dir, sources_path)
        source, _ = sf.read(sources_path, dtype='float32')
        if max_length < len(source):
            max_length = len(source)
        sources_list.append(source)

    noise_path = os.path.join(wham_dir, row['noise_path'])
    noise, _ = sf.read(noise_path, dtype='float32', stop=max_length)
    if len(noise.shape) > 1:
        noise = noise[:, 0]
    if len(noise) < max_length:
        noise = extend_noise(noise, max_length)
    sources_list.append(noise)
    gain_list.append(row['noise_gain'])

    return mixture_id, gain_list, sources_list


def get_list_from_csv(row, column, n_src):
    python_list = []
    for i in range(n_src):
        current_column = column.split('_')
        current_column.insert(1, str(i + 1))
        current_column = '_'.join(current_column)
        python_list.append(row[current_column])
    return python_list


def extend_noise(noise, max_length):
    noise_ex = noise
    window = np.hanning(RATE + 1)
    i_w = window[:len(window) // 2 + 1]
    d_w = window[len(window) // 2::-1]
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate((noise_ex[:len(noise_ex) - len(d_w)],
                                   np.multiply(noise_ex[len(noise_ex) - len(d_w):], d_w) +
                                   np.multiply(noise[:len(i_w)], i_w),
                                   noise[len(i_w):]))
    noise_ex = noise_ex[:max_length]
    return noise_ex


def transform_sources(sources_list, freq, mode, gain_list):
    sources_list_norm = loudness_normalize(sources_list, gain_list)
    sources_list_resampled = resample_list(sources_list_norm, freq)
    reshaped_sources = fit_lengths(sources_list_resampled, mode)
    return reshaped_sources


def loudness_normalize(sources_list, gain_list):
    return [s * g for s, g in zip(sources_list, gain_list)]


def resample_list(sources_list, freq):
    return [resample_poly(source, freq, RATE) for source in sources_list]


def fit_lengths(source_list, mode):
    if mode == 'min':
        target_length = min(len(source) for source in source_list)
        return [source[:target_length] for source in source_list]
    else:
        target_length = max(len(source) for source in source_list)
        return [np.pad(source, (0, target_length - len(source)), mode='constant')
                for source in source_list]


def mix(sources_list):
    mixture = np.zeros_like(sources_list[0])
    for source in sources_list:
        mixture += source
    return mixture


def compute_snr_list(mixture, sources_list):
    return [snr_xy(s, mixture - s) for s in sources_list]


def snr_xy(x, y):
    return 10 * np.log10(np.mean(x ** 2) / (np.mean(y ** 2) + EPS) + EPS)


def add_to_metrics_metadata(metrics_df, mixture_id, snr_list):
    row_metrics = [mixture_id] + snr_list
    metrics_df.loc[len(metrics_df)] = row_metrics


def add_to_mixture_metadata(mix_df, mix_id, abs_mix_path, abs_sources_path,
                            abs_noise_path, length, subdir):
    sources_path = abs_sources_path
    noise_path = [abs_noise_path]
    if subdir == 'mix_clean':
        noise_path = []
    elif subdir == 'mix_single':
        sources_path = [abs_sources_path[0]]
    row_mixture = [mix_id, abs_mix_path] + sources_path + noise_path + [length]
    mix_df.loc[len(mix_df)] = row_mixture


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
