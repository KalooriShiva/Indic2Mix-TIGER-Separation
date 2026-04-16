import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm
from rich import print


def preprocess_one_dir(in_data_dir, out_dir, data_type):
    """Create .json file for one condition."""
    mix_infos = []
    s1_infos = []
    s2_infos = []
    in_dir = os.path.abspath(os.path.join(in_data_dir, data_type))
    print("Process {} set...".format(data_type))
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith(".wav") :
                file_path = os.path.join(root, file)
                audio, _ = sf.read(file_path)
                mix_infos.append((
                    file_path,
                    len(audio),
                ))
                
                file_path = file_path.replace("mix_both", "s1")
                audio, _ = sf.read(file_path)
                s1_infos.append((
                    file_path,
                    len(audio),
                ))
                
                file_path = file_path.replace("s1", "s2")
                audio, _ = sf.read(file_path)
                s2_infos.append((
                    file_path,
                    len(audio),
                ))
                print("Process num: {}".format(len(mix_infos)), end="\r")
                
    if not os.path.exists(os.path.join(out_dir, data_type)):
        os.makedirs(os.path.join(out_dir, data_type))
    with open(os.path.join(out_dir, data_type, "mix.json"), "w") as f:
        json.dump(mix_infos, f, indent=4)
    
    with open(os.path.join(out_dir, data_type, "s1.json"), "w") as f:
        json.dump(s1_infos, f, indent=4)
        
    with open(os.path.join(out_dir, data_type, "s2.json"), "w") as f:
        json.dump(s2_infos, f, indent=4)


def preprocess_lrs2_audio(inp_args):
    """Create .json files for all conditions."""
    for data_type in ["tr/mix_both", "cv/mix_both", "tt/mix_both"]:
        preprocess_one_dir(
            inp_args.in_dir, inp_args.out_dir, data_type
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LRS2 audio data preprocessing")
    parser.add_argument(
        "--in_dir",
        type=str,
        default="/home/kaloori.shiva/Thesis/TIGER/openSLR_hin_dataset/freesounds/Indic2mix/Telugu/kathbath_freesounds_2mix_output/Libri2Mix/wav16k/min",
        help="Directory path of audio including tr, cv and tt",
    )
    parser.add_argument(
        "--out_dir", type=str, default="/home/kaloori.shiva/Thesis/TIGER/DataPreProcess/Indic2mix/Telugu", help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess_lrs2_audio(args)
 
