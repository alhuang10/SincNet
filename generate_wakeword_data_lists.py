import os
import numpy as np
import random
from collections import Counter, defaultdict

random.seed(123)

OUTPUT_DIR = 'wakeword_file_lists'

UTTERANCE_COUNT_THRESHOLD = 10  # Min utterances for a speaker to be used
UTTERANCE_TRAIN_COUNT = 6

ENROLLMENT_TRAIN_COUNT = 447

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


def get_hey_webex_balanced(file_set):
    """
    Gets filepaths of all files that have hey_webex
    """
    hey_webex_files = [x for x in file_set if 'hey_webex' in x and ".wav" in x]

    # Get the speaker ID which is 2-letter initial followed by 3-digit number
    speaker_ids = [x.split('-')[1] for x in hey_webex_files]

    # Organize files by speaker
    speaker_id_to_files = defaultdict(list)
    for f in hey_webex_files:
        speaker = f.split('-')[1]
        speaker_id_to_files[speaker].append(f)

    # For each speaker with 10 or more utterances split into train and test
    hey_webex_train = []
    hey_webex_test = []

    for id, files in speaker_id_to_files.items():
        if len(files) >= UTTERANCE_COUNT_THRESHOLD:
            # Shuffle files
            random.shuffle(files)

            selected_files = files[:UTTERANCE_COUNT_THRESHOLD]

            train_split = selected_files[:UTTERANCE_TRAIN_COUNT]
            test_split = selected_files[UTTERANCE_TRAIN_COUNT:]

            hey_webex_train.extend(train_split)
            hey_webex_test .extend(test_split)

    with open(os.path.join(OUTPUT_DIR, 'hey_webex_all.txt'), 'w') as f:
        for item in hey_webex_files:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'hey_webex_train.txt'), 'w') as f:
        for item in hey_webex_train:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'hey_webex_test.txt'), 'w') as f:
        for item in hey_webex_test:
            f.write("%s\n" % item)

    selected_files = hey_webex_train + hey_webex_test
    generate_label_dict(hey_webex_files, "hey_webex_label_dict.npy")


def get_okay_webex_balanced(file_set):
    """
    Gets filepaths of all files that have okay_webex
    """
    okay_webex_files = [x for x in file_set if 'okay_webex' in x and ".wav" in x]

    # Get the speaker ID which is 2-letter initial followed by 3-digit number
    speaker_ids = [x.split('-')[1] for x in okay_webex_files]

    # Organize files by speaker
    speaker_id_to_files = defaultdict(list)
    for f in okay_webex_files:
        speaker = f.split('-')[1]
        speaker_id_to_files[speaker].append(f)

    # For each speaker with 10 or more utterances split into train and test
    okay_webex_train = []
    okay_webex_test = []

    for id, files in speaker_id_to_files.items():
        if len(files) >= UTTERANCE_COUNT_THRESHOLD:
            # Shuffle files
            random.shuffle(files)

            selected_files = files[:UTTERANCE_COUNT_THRESHOLD]

            train_split = selected_files[:UTTERANCE_TRAIN_COUNT]
            test_split = selected_files[UTTERANCE_TRAIN_COUNT:]

            okay_webex_train.extend(train_split)
            okay_webex_test.extend(test_split)

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_all.txt'), 'w') as f:
        for item in okay_webex_files:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_train.txt'), 'w') as f:
        for item in okay_webex_train:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_test.txt'), 'w') as f:
        for item in okay_webex_test:
            f.write("%s\n" % item)

    selected_files = okay_webex_train + okay_webex_test
    generate_label_dict(selected_files, "okay_webex_label_dict.npy")
    print(get_unique_items(os.path.join(OUTPUT_DIR, "okay_webex_label_dict.npy")))

def get_okay_webex_enrollment_data_lists(file_set):

    okay_webex_files = [x for x in file_set if 'okay_webex' in x and ".wav" in x]
    speaker_ids = [x.split('-')[1] for x in okay_webex_files]

    # Organize files by speaker
    speaker_id_to_files = defaultdict(list)
    for f in okay_webex_files:
        speaker = f.split('-')[1]
        speaker_id_to_files[speaker].append(f)

    valid_speaker_ids = [speaker_id for speaker_id, files in speaker_id_to_files.items()
                         if len(files) > UTTERANCE_COUNT_THRESHOLD]

    random.shuffle(valid_speaker_ids)

    # Separate training and test speakers
    enrollment_train_ids = set(valid_speaker_ids[:ENROLLMENT_TRAIN_COUNT])
    enrollment_test_ids = set(valid_speaker_ids[ENROLLMENT_TRAIN_COUNT:])

    # For each speaker in enrollment_train, 6 queries go to training and 4 queries go to dev for parameter optimization
    train_enrollment = []
    dev_enrollment = []

    # For each speaker in test, 6 queries used to generate the d-vector and 4 used for testing (nearest neighbor)
    test_queries_seen = []
    test_queries_unseen = []

    for id in enrollment_train_ids:
        files = speaker_id_to_files[id]

        random.shuffle(files)
        selected_files = files[:UTTERANCE_COUNT_THRESHOLD]
        train_split = selected_files[:UTTERANCE_TRAIN_COUNT]
        test_split = selected_files[UTTERANCE_TRAIN_COUNT:]

        train_enrollment.extend(train_split)
        dev_enrollment.extend(test_split)

    for id in enrollment_test_ids:
        files = speaker_id_to_files[id]

        random.shuffle(files)
        selected_files = files[:UTTERANCE_COUNT_THRESHOLD]
        train_split = selected_files[:UTTERANCE_TRAIN_COUNT]
        test_split = selected_files[UTTERANCE_TRAIN_COUNT:]

        test_queries_seen.extend(train_split)
        test_queries_unseen.extend(test_split)

    print(len(train_enrollment), len(dev_enrollment), len(test_queries_seen), len(test_queries_unseen))

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_enrollment_train.txt'), 'w') as f:
        for item in train_enrollment:
            f.write("%s\n" % item)

    with open(os.path.join(OUTPUT_DIR, 'okay_webex_enrollment_dev.txt'), 'w') as f:
        for item in dev_enrollment:
            f.write("%s\n" % item)

    # Queries from users that the network does not train with, used to generate a corresponding d-vector
    with open(os.path.join(OUTPUT_DIR, 'okay_webex_enrollment_test_seen.txt'), 'w') as f:
        for item in test_queries_seen:
            f.write("%s\n" % item)

    # Queries used for evaluation on d-vectors
    with open(os.path.join(OUTPUT_DIR, 'okay_webex_enrollment_test_unseen.txt'), 'w') as f:
        for item in test_queries_unseen:
            f.write("%s\n" % item)

    selected_files = train_enrollment + dev_enrollment
    generate_label_dict(selected_files, "okay_webex_label_dict_enrollment.npy")
    print(get_unique_items(os.path.join(OUTPUT_DIR, "okay_webex_label_dict_enrollment.npy")))


def get_unique_items(label_dict_file):
    label_dict = np.load(label_dict_file).item()
    return len(set(list(label_dict.values())))


def generate_label_dict(file_set, dict_name):

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

    np.save(os.path.join(OUTPUT_DIR, dict_name), label_dict)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    file_set = get_all_files_recursive('/mnt/extradrive2/wakeword_data/')

    get_okay_webex_balanced(file_set)
    # get_hey_webex_balanced(file_set)

    # get_okay_webex_enrollment_data_lists(file_set)