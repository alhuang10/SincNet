import os
import numpy as np
import random
import pickle
from collections import defaultdict, OrderedDict

OUTPUT_DIR = 'wakeword_file_lists'


def get_unique_items(label_dict_file):
    label_dict = np.load(label_dict_file).item()
    return len(set(list(label_dict.values())))


def generate_label_dict(file_set, output_dir, dict_name):

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

    np.save(os.path.join(output_dir, dict_name), label_dict)


def get_all_files_recursive(root_dir):
    """
    Recursively walk through the dataset directory and return full filepaths to each file as a sorted list.
    Args:
        root_dir: The directory with all sound files.

    Returns: Sorted list of all filepaths to sound files

    """
    file_set = set()
    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            relative_dir = os.path.relpath(dir_, root_dir)
            relative_file = os.path.join(relative_dir, file_name)

            file_set.add(relative_file)

    file_set = list(file_set)
    file_set.sort()

    return file_set


def get_relevant_sound_files(file_set, utterance_text):
    """
    Get's all the sound files of users mentioning an utterance text
    Args:
        file_set: The list of all sound files, filepaths to each file
        utterance_text: The text we want the sound files to contain

    Returns: A list of all the files with the utterance

    """
    return [x for x in file_set if utterance_text in x and ".wav" in x]


def get_speaker_to_files_mapping(sound_file_list):
    """
    Gets a mapping from speaker ID (eg. SV101) to all the sound files corresponding to that speaker
    Args:
        sound_file_list: List of all sound files we want to consider, typically all files containing a given utterance

    Returns:
        A dictionary mapping speaker ID to all sound files of that speaker saying a given text
    """
    # Organize files by speaker
    speaker_id_to_files = defaultdict(list)
    for f in sound_file_list:
        speaker = f.split('-')[1]
        speaker_id_to_files[speaker].append(f)

    return speaker_id_to_files


def get_softmax_training_data_lists(file_set, utterance_text, query_count_threshold, num_train_files_per_speaker):
    """
    Get a train/test data list split given text for queries to contain, and appropriate thresholds
    Args:
        file_set: all sound files in the dataset
        utterance_text: text that the sound queries should contain
        query_count_threshold: number of queries that each speaker must have for them to be included in training
        num_train_files_per_speaker: Given the "query_count_threshold" queries per speaker, number of files to use for
                                     training and number of files to use for dev
    Returns:
        N/A, generates data lists and label dictionary
    """
    # Directory name includes threshold/training counts
    specific_output_directory = os.path.join(OUTPUT_DIR,
                                             f"{utterance_text}_softmax_{query_count_threshold}_query_threshold_" +
                                             f"{num_train_files_per_speaker}_train_count")

    if not os.path.exists(specific_output_directory):
        os.mkdir(specific_output_directory)

    sound_files_containing_text = get_relevant_sound_files(file_set, utterance_text)
    speaker_id_to_files = get_speaker_to_files_mapping(sound_files_containing_text)

    valid_speaker_ids = [speaker_id for speaker_id, files in speaker_id_to_files.items()
                         if len(files) >= query_count_threshold]
    print("Number of speakers with required number of utterances:", len(valid_speaker_ids))

    train_files = []
    dev_files = []

    for _, files in speaker_id_to_files.items():
        if len(files) >= query_count_threshold:
            # Shuffle files and select files equal to "query_count_threshold"
            random.shuffle(files)
            selected_files = files[:query_count_threshold]

            # Divide the selected files into train/dev
            train_split = selected_files[:num_train_files_per_speaker]
            dev_split = selected_files[num_train_files_per_speaker:]

            train_files.extend(train_split)
            dev_files.extend(dev_split)

    with open(os.path.join(specific_output_directory, "all.txt"), 'w') as f:
        for item in sound_files_containing_text:
            f.write("%s\n" % item)

    with open(os.path.join(specific_output_directory, "train.txt"), 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)

    with open(os.path.join(specific_output_directory, "dev.txt"), 'w') as f:
        for item in dev_files:
            f.write("%s\n" % item)

    print(f"Number of train files: {len(train_files)}, Number of dev files: {len(dev_files)}")

    # Generate the label dictionary for the files/speakers selected
    selected_files = train_files + dev_files
    generate_label_dict(selected_files, specific_output_directory, "label_dict.npy")


def get_enrollment_training_data_lists(file_set, utterance_text, query_count_threshold, num_train_files_per_speaker,
                                       train_vs_enrollment_fraction):
    """
    Get an training/enrollment data split as well as splitting the training list into TRAIN/DEV and the enrollment
    list into ENROLL/TEST
    Args:
        file_set: all sound files in the dataset
        utterance_text: text that the sound queries should contain
        query_count_threshold: number of queries that each speaker must have for them to be included in training
        num_train_files_per_speaker: Given the "query_count_threshold" queries per speaker, number of files to use for
                                     training and number of files to use for dev
        train_vs_enrollment_fraction: The fraction of total speakers to use for training, other portion will be used
                                      for enrollment vector generation/testing
    Returns:
        N/A, generates data lists and label dictionary
    """
    # Directory name includes threshold/training counts
    specific_output_directory = os.path.join(OUTPUT_DIR,
                                             f"{utterance_text}_enrollment_{query_count_threshold}_query_threshold_" +
                                             f"{num_train_files_per_speaker}_train_count")

    if not os.path.exists(specific_output_directory):
        os.mkdir(specific_output_directory)

    sound_files_containing_text = get_relevant_sound_files(file_set, utterance_text)
    speaker_id_to_files = get_speaker_to_files_mapping(sound_files_containing_text)

    valid_speaker_ids = [speaker_id for speaker_id, files in speaker_id_to_files.items()
                         if len(files) >= query_count_threshold]
    training_count = int(len(valid_speaker_ids)*train_vs_enrollment_fraction)

    print(f"Number of training speakers: {training_count}")
    print(f"Number of enrollment speakers: {len(valid_speaker_ids) - training_count}")
    print(f"Number of total speakers with enough utterances: {len(valid_speaker_ids)}")

    random.shuffle(valid_speaker_ids)
    valid_speaker_ids = list(OrderedDict.fromkeys(valid_speaker_ids))

    # Separate training and test speakers
    enrollment_train_ids = valid_speaker_ids[:training_count]
    enrollment_test_ids = valid_speaker_ids[training_count:]

    print("First enrollment train ID:", enrollment_train_ids[0])
    print("First enrollment test ID:", enrollment_test_ids[0])

    # For each speaker in enrollment_train, some queries go to training
    # and some queries go to dev for parameter optimization
    train_files = []
    dev_files = []

    # For each speaker in test, some queries go to generate the d-vector
    # and some go for testing (nearest neighbor)
    test_queries_seen = []
    test_queries_unseen = []

    for speaker_id in enrollment_train_ids:
        files = speaker_id_to_files[speaker_id]

        random.shuffle(files)
        selected_files = files[:query_count_threshold]
        train_split = selected_files[:num_train_files_per_speaker]
        test_split = selected_files[num_train_files_per_speaker:]

        train_files.extend(train_split)
        dev_files.extend(test_split)

    for speaker_id in enrollment_test_ids:
        files = speaker_id_to_files[speaker_id]

        random.shuffle(files)
        selected_files = files[:query_count_threshold]
        train_split = selected_files[:num_train_files_per_speaker]
        test_split = selected_files[num_train_files_per_speaker:]

        test_queries_seen.extend(train_split)
        test_queries_unseen.extend(test_split)

    print("Training, dev, enrollment_length, enrollment_test")
    print(len(train_files), len(dev_files), len(test_queries_seen), len(test_queries_unseen))

    with open(os.path.join(specific_output_directory, "enrollment_train.txt"), 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)

    with open(os.path.join(specific_output_directory, "enrollment_dev.txt"), 'w') as f:
        for item in dev_files:
            f.write("%s\n" % item)

    # Queries from users that the network does not train with, used to generate a corresponding d-vector
    with open(os.path.join(specific_output_directory, "enrollment_test_seen.txt"), 'w') as f:
        for item in test_queries_seen:
            f.write("%s\n" % item)

    # Queries used for evaluation on d-vectors
    with open(os.path.join(specific_output_directory, "enrollment_test_unseen.txt"), 'w') as f:
        for item in test_queries_unseen:
            f.write("%s\n" % item)

    selected_files = train_files + dev_files
    generate_label_dict(selected_files, specific_output_directory, "label_dict_enrollment.npy")

    selected_test_files = test_queries_seen + test_queries_unseen
    generate_label_dict(selected_test_files, specific_output_directory, "label_dict_enrollment_test.npy")

    # Save the training and enrollment users
    with open(os.path.join(specific_output_directory, "enrollment_train_ids.p"), 'wb') as f:
        pickle.dump(enrollment_train_ids, f)
    with open(os.path.join(specific_output_directory, "enrollment_test_ids.p"), 'wb') as f:
        pickle.dump(enrollment_test_ids, f)

    print("Training speakers",
          get_unique_items(os.path.join(specific_output_directory, "label_dict_enrollment.npy")))
    print("Enroll speakers",
          get_unique_items(os.path.join(specific_output_directory, "label_dict_enrollment_test.npy")))


def generate_enrollment_list(file_set, utterance_text,
                             pickle_filepath, unseen, count_threshold, train_count, num_unique_ids):
    """
    Given a pickle file of speaker ids, generate enrollment/test lists.
    Used for testing enrollment with different ratios than used in training while controlling for using speakers
    that the model has seen/not seen.
    Args:
        file_set:
        utterance_text:
        pickle_filepath:
        unseen:
        count_threshold:
        train_count:
        num_unique_ids:

    Returns:

    """
    speaker_id_list = pickle.load(open(pickle_filepath, 'rb'))

    if unseen:
        unseen_str = "unseen"
    else:
        unseen_str = "seen"

    # Directory name includes threshold/training counts
    specific_output_directory = \
        os.path.join(OUTPUT_DIR,
                     f"only_enrollment_{utterance_text}_{count_threshold}_" +
                     f"threshold_{train_count}_train_count_{num_unique_ids}_unique_speakers_{unseen_str}")

    if not os.path.exists(specific_output_directory):
        os.mkdir(specific_output_directory)

    sound_files_containing_text = get_relevant_sound_files(file_set, utterance_text)
    speaker_id_to_files = get_speaker_to_files_mapping(sound_files_containing_text)

    valid_speaker_ids = [speaker_id for speaker_id, files in speaker_id_to_files.items()
                         if len(files) >= count_threshold and speaker_id in speaker_id_list]

    # Only use a given number of unique speakers
    if num_unique_ids > len(valid_speaker_ids):
        print("Too many unique ids requested, using all unique speakers available:", len(valid_speaker_ids))
    else:
        valid_speaker_ids = valid_speaker_ids[:num_unique_ids]

    test_queries_seen = []
    test_queries_unseen = []

    for speaker_id in valid_speaker_ids:
        files = speaker_id_to_files[speaker_id]

        random.shuffle(files)
        selected_files = files[:count_threshold]
        train_split = selected_files[:train_count]
        test_split = selected_files[train_count:]

        test_queries_seen.extend(train_split)
        test_queries_unseen.extend(test_split)

    print(f"Num queries for enrollment: {len(test_queries_seen)}, Num queries for test: {len(test_queries_unseen)}")

    # Queries from users that the network does not train with, used to generate a corresponding d-vector
    with open(os.path.join(specific_output_directory, 'enrollment_test_seen.txt'), 'w') as f:
        for item in test_queries_seen:
            f.write("%s\n" % item)

    # Queries used for evaluation on d-vectors
    with open(os.path.join(specific_output_directory, 'enrollment_test_unseen.txt'), 'w') as f:
        for item in test_queries_unseen:
            f.write("%s\n" % item)

    # Create label dict
    selected_test_files = test_queries_seen + test_queries_unseen
    generate_label_dict(selected_test_files, specific_output_directory, "label_dict_enrollment_test.npy")


if __name__ == '__main__':
    UTTERANCE_COUNT_THRESHOLD = 30  # Min utterances for a speaker to be used
    UTTERANCE_TRAIN_COUNT = 25
    ENROLLMENT_TRAIN_FRACTION = 0.8
    random.seed(UTTERANCE_COUNT_THRESHOLD*UTTERANCE_TRAIN_COUNT)

    file_set = get_all_files_recursive('/mnt/extradrive2/wakeword_data/')

    # get_softmax_training_data_lists(file_set, "okay_webex", 30, 25)
    # get_enrollment_training_data_lists(file_set, "okay_webex", 30, 25, .8)

    generate_enrollment_list(file_set,
                             "okay_webex",
                             "wakeword_file_lists/enrollment_30_threshold_25_train_count/enrollment_test_ids.p",
                             "True",
                             30,
                             25,
                             30)
