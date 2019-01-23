import os
import numpy as np


OUTPUT_DIR = 'wakeword_file_lists'


def get_all_files_recursive(root_dir):
    file_set = set()
    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            relative_dir = os.path.relpath(dir_, root_dir)
            relative_file = os.path.join(relative_dir, file_name)

            file_set.add(relative_file)

    file_set = list(file_set)
    file_set.sort()

    return file_set


def get_hey_webex(file_set):
    """
    Gets filepaths of all files that have hey_webex
    """
    hey_webex_files = [x for x in file_set if 'hey_webex' in x and ".wav" in x]
    hey_webex_train = [x for x in hey_webex_files if 'train/' in x]
    hey_webex_test = [x for x in hey_webex_files if 'test/' in x]

    with open(os.path.join(OUTPUT_DIR, 'hey_webex_all.txt'), 'w') as f:
        for item in hey_webex_files:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'hey_webex_train.txt'), 'w') as f:
        for item in hey_webex_train:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'hey_webex_test.txt'), 'w') as f:
        for item in hey_webex_test:
            f.write("%s\n" % item)


def get_okay_webex(file_set):
    """
    Gets filepaths of all files that have okay_webex
    """
    okay_webex_files = [x for x in file_set if 'okay_webex' in x and ".wav" in x]
    okay_webex_train = [x for x in okay_webex_files if 'train/' in x]
    okay_webex_test = [x for x in okay_webex_files if 'test/' in x]

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_all.txt'), 'w') as f:
        for item in okay_webex_files:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_train.txt'), 'w') as f:
        for item in okay_webex_train:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_test.txt'), 'w') as f:
        for item in okay_webex_test:
            f.write("%s\n" % item)


def genererate_label_dict(file_set):

    label_dict = {}

    speaker_to_id = {}
    current_count = 0

    for f in file_set:
        if f[-4:] == ".wav":
            speaker_id = f.split('-')[1]
            if speaker_id not in speaker_to_id:
                label_dict[f] = current_count

                speaker_to_id[speaker_id] = current_count
                current_count += 1
            else:
                label_dict[f] = speaker_to_id[speaker_id]

    np.save("wakeword_file_lists/wakeword_labels.npy", label_dict)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    file_set = get_all_files_recursive('/mnt/extradrive2/wakeword_data/')

    get_okay_webex(file_set)
    get_hey_webex(file_set)

    genererate_label_dict(file_set)