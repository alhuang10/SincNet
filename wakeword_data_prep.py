import shutil
import os
import soundfile as sf
import numpy as np
import sys
import pandas as pd

from generate_vad_mapping import get_vad_mapping

CSV_PATHS = ['wakeword_data/train/time_align_20180306.csv',
             'wakeword_data/train/time_align_20180319.csv',
             'wakeword_data/train/time_align_20180402.csv']


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def copy_folder(in_folder, out_folder):
    if os.path.isdir(out_folder):
        print("Out folder already exists, not copying structure")
    else:
        shutil.copytree(in_folder, out_folder, ignore=ig_f)


if __name__ == '__main__':
    missing_count = 0
    valid_count = 0

    vad_mapping = get_vad_mapping(CSV_PATHS)

    in_folder = sys.argv[1]  # Raw data sample folder
    out_folder = sys.argv[2]  # Normalized and pared samples
    file_list = sys.argv[3]  # Path to file containing file paths to .wav sound files
    # Replicate input folder structure to output folder
    copy_folder(in_folder, out_folder)

    with open(file_list, 'r') as f:
        filepaths = f.read().splitlines()

    for filepath in filepaths:

        raw_filepath = os.path.join(in_folder, filepath)
        [signal, sample_rate] = sf.read(raw_filepath)
        signal = signal.astype(np.float64)

        # Signal normalization
        signal = signal / np.abs(np.max(signal))

        filename = filepath.split('/')[-1]

        if filename not in vad_mapping:
            missing_count += 1
        else:
            # Approximate start and end of the audio
            start_ind, end_ind = vad_mapping[filename]
            start_ind = int((start_ind/1000.0) * sample_rate)
            end_ind = int((end_ind/1000.0) * sample_rate)

            # Remove silences
            signal = signal[start_ind:end_ind]

            normalized_filepath = os.path.join(out_folder, filepath)
            sf.write(normalized_filepath, signal, sample_rate)

            valid_count += 1

        print(f"Valid: {valid_count}, Missing: {missing_count}")
