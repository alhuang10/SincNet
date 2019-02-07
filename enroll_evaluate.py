import os
# import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import time
import sys
from collections import defaultdict
import pickle
import numpy as np
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool


def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, label_dict, fact_amp):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    for i in range(batch_size):
        # select a random sentence from the list
        # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768

        [signal, fs] = sf.read(os.path.join(data_folder, wav_lst[snt_id_arr[i]]))

        # accessing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen

        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        lab_batch[i] = label_dict[wav_lst[snt_id_arr[i]]]

    inp = Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
    lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

    return inp, lab


use_voting = False

# Reading cfg file
options = read_conf()

# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder
output_folder = options.output_folder
tracking_file = options.tracking_file

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))
fc_return_hidden_layer = str_to_bool(options.fc_return_hidden_layer)

# [class]
class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# training list
train_files = ReadList(tr_lst)
snt_tr = len(train_files)

# test list
test_files = ReadList(te_lst)
snt_te = len(test_files)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop,
            }

CNN_net = CNN(CNN_arch)
CNN_net.cuda()

# Loading label dictionary
label_dict = np.load(class_dict_file).item()

# assert class_lay[0] == len(set(list(label_dict.values()))), \
#     f"class_lay param must be equal to number of unique speakers: {len(set(list(label_dict.values())))}"

DNN1_arch = {'input_dim': CNN_net.out_dim,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             'fc_return_hidden_layer': fc_return_hidden_layer
             }

DNN1_net = MLP(DNN1_arch)
DNN1_net.cuda()

DNN2_arch = {'input_dim': fc_lay[-1],
             'fc_lay': class_lay,
             'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm,
             'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp,
             'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act,
             }

DNN2_net = MLP(DNN2_arch)
DNN2_net.cuda()

if pt_file != 'none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])


# Enrollment/d-vector generation portion

# Mapping from integer label to the generated d-vector
label_to_vector_mapping = defaultdict(list)

with torch.no_grad():
    for i, sound_file in enumerate(train_files):
        print("Train:", i, len(train_files))

        [signal, fs] = sf.read(os.path.join(data_folder, sound_file))

        signal = torch.from_numpy(signal).float().cuda().contiguous()
        label = label_dict[sound_file]

        # split signals into chunks
        beg_samp = 0
        end_samp = wlen

        sig_arr = torch.zeros([Batch_dev, wlen], dtype=torch.float32, device=torch.device('cuda:0'))
        count_fr = 0
        count_fr_tot = 0

        while end_samp < signal.shape[0]:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp + wshift
            end_samp = beg_samp + wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1

            if count_fr == Batch_dev:
                inp = Variable(sig_arr.contiguous())
                count_fr = 0

        if count_fr > 0:
            inp = Variable(sig_arr[0:count_fr].contiguous())

        CNN_output = CNN_net(inp)
        dnn1_output, final_hidden_state = DNN1_net(CNN_output)

        # For each unique speaker, keep track of the vector rep for each utterance
        hidden_state_vector_rep = torch.mean(final_hidden_state, dim=0).cpu().numpy()
        label_to_vector_mapping[label].append(hidden_state_vector_rep)

if not use_voting:
    # Average the utterance vectors to get a final representation
    for i, vectors in label_to_vector_mapping.items():
        label_to_vector_mapping[i] = sum(vectors) / len(vectors)

# pickle.dump(label_to_vector_mapping, open("label_to_vector_mapping.p", 'wb'))


def get_nearest_neighbor_label(hidden_state, label_to_vector):
    lowest_distance = float("inf")
    best_label = -1

    for label, vector in label_to_vector.items():
        distance = np.linalg.norm(vector - hidden_state)

        if distance < lowest_distance:
            lowest_distance = distance
            best_label = label

    return best_label


def get_nearest_neighbor_label_voting(hidden_state, label_to_vector_list):
    lowest_distance = float("inf")
    best_label = -1

    for label, vector_list in label_to_vector_list.items():
        distances = [np.linalg.norm(vector - hidden_state) for vector in vector_list]

        # distance = sum([distances])
        distance = sum([x*x for x in distances])  # Use squared differences

        if distance < lowest_distance:
            lowest_distance = distance
            best_label = label

    return best_label


# Testing period
correct = 0
incorrect = 0

with torch.no_grad():
    for i, sound_file in enumerate(test_files):
        [signal, fs] = sf.read(os.path.join(data_folder, sound_file))

        signal = torch.from_numpy(signal).float().cuda().contiguous()
        actual_label = label_dict[sound_file]

        # split signals into chunks
        beg_samp = 0
        end_samp = wlen

        sig_arr = torch.zeros([Batch_dev, wlen], dtype=torch.float32, device=torch.device('cuda:0'))
        count_fr = 0
        count_fr_tot = 0

        while end_samp < signal.shape[0]:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp + wshift
            end_samp = beg_samp + wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1

            if count_fr == Batch_dev:
                inp = Variable(sig_arr.contiguous())
                count_fr = 0

        if count_fr > 0:
            inp = Variable(sig_arr[0:count_fr].contiguous())

        CNN_output = CNN_net(inp)
        dnn1_output, final_hidden_state = DNN1_net(CNN_output)

        hidden_state_vector_rep = torch.mean(final_hidden_state, dim=0).cpu().numpy()

        # Get the nearest neighbor for the final hidden state to choose the label
        if use_voting:
            label_choice = get_nearest_neighbor_label_voting(hidden_state_vector_rep, label_to_vector_mapping)
        else:
            label_choice = get_nearest_neighbor_label(hidden_state_vector_rep, label_to_vector_mapping)

        if label_choice == actual_label:
            correct += 1
        else:
            incorrect += 1

        print("Test:", i, len(test_files))

    print("Correct:", correct)
    print("Incorrect:", incorrect)


