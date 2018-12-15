from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import Utils
import LSTMTrain

tf.logging.set_verbosity(tf.logging.DEBUG)

SIZEEMBEDDING = 4
SEQUENCELENGTH = 100


# targeted_protein = 1
# TRAIN_DIR='./tmp'+str(targeted_protein)+'/'

def main(unused_argv):
    # global targeted_protein
    targeted_protein_start = int(unused_argv[1])
    # Load training and eval data
    protein_seq_dict = Utils.load_obj('protein_seq_dict_60')
    # print(sorted(protein_seq_dict))
    labels_original = []
    data = []
    label = 0
    len_sequences = 0
    # protein_seq_dict_sorted = sorted(protein_seq_dict)
    for family in sorted(protein_seq_dict):
        for sequences in protein_seq_dict[family]:

            flat_list = [item for sublist in sequences for item in sublist]
            if len(flat_list) % (SEQUENCELENGTH * SIZEEMBEDDING) == 0:
                data.append(flat_list)
                len_sequences += len(flat_list)
                labels_original.append(label)
                # if label == CNNComplex.FLAGS.protein_no:
                #     labels.append(1)
                # else:
                #     labels.append(0)
                # data.append(np.array(flat_list,dtype=np.float16))
                # labels.append(1 if label==protein_no else 0)
        label += 1

    labels_original = np.asarray(labels_original, dtype=np.int32)
    # print(len(protein_seq_dict))
    for start_from in range(targeted_protein_start, 2298):
        pr_name = sorted(protein_seq_dict.keys())[start_from]
        print("currently considering " + pr_name)
        labels = np.zeros(len(labels_original))
        unique, counts = np.unique(labels_original, return_counts=True)
        counts_seq = dict(zip(unique, counts))
        num_seq = 0
        if start_from in counts_seq:  # .has_key(start_from):
            num_seq = counts_seq[start_from]
        if num_seq < 10:
            print("Too few (<10) sequences for " + pr_name)
            continue
        start = 0
        for i in range(0, start_from):
            if i in counts_seq:  # .has_key(i):
                start += counts_seq[i]
        for i in range(0, len(labels_original)):
            labels[i] = 1 if labels_original[i] == start_from else 0
        # index = np.argmax(labels > 0)
        labels = np.asarray(labels, dtype=np.int32)
        data = np.asarray(data, dtype=np.float32)
        data = np.reshape(data, (len(labels), SEQUENCELENGTH, SIZEEMBEDDING))
        # data = np.reshape(data,(len(labels),SIZEEMBEDDING*60))
        # data = np.reshape(data,(len(labels)*len(sequences)*SIZEEMBEDDING,1))
        assert data.shape[0] == labels.shape[0]

        data_must = data[start:num_seq, :]
        label_must = labels[start:num_seq, ]
        # data = np.reshape(np.array(data), (len_sequences*SIZEEMBEDDING*len(labels),1))
        # x_ph = tf.placeholder(tf.float32, shape=[len(protein_seq_dict),None, 8])
        # data = tf.data.Dataset.from_tensor_slices(data)
        # labels= tf.data.Dataset.from_tensor_slices(labels)
        # dataset = tf.data.Dataset.zip((data,labels ))
        np.random.seed(100)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]
        labels = np.asarray(labels, dtype=np.int32)
        train_dir = './LSTM_' + pr_name + '/'
        print(train_dir)
        # CNNtrain.train(data, labels, label,data_must,label_must,train_dir)
        LSTMTrain.train(data, labels, data_must, label_must, train_dir)

        break

if __name__ == "__main__":
    tf.app.run()
