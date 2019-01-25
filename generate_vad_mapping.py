import pandas as pd


def get_vad_mapping(csv_paths):
    # Mapping from filename to a tuple of (start_vad, end_vad)
    vad_mapping = {}

    for time_align_filepath in csv_paths:
        time_align_df = pd.read_csv(open(time_align_filepath, 'r'), sep="\t")

        filenames = list(time_align_df.filename)
        start_frames = list(time_align_df.start_0_vad)
        end_frames = list(time_align_df.end_1_vad)

        for f, start, end in zip(filenames, start_frames, end_frames):
            # Some starts are negative, which messes up signal slicing
            start = max(0, start)

            if f not in vad_mapping:
                vad_mapping[f] = (start, end)

    return vad_mapping
