from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import Utils
import numpy as np
import tensorflow as tf
import LSTMProteinSequenceMain
import LSTMComplex
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix

tf.logging.set_verbosity(tf.logging.DEBUG)
NUM_CLASSES = 2
ProteinName = 'PF01993'
fmainDir = './store/LSTM_'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size_test', 128,
                            """Number of hidden layer""")
tf.app.flags.DEFINE_integer('hidden_layer2', 64,
                            """Number of hidden layer""")
tf.app.flags.DEFINE_string('eval_dir', fmainDir + ProteinName + '/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('start', 0,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_examples', 500,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(cdir, saver, top_k_op, labels_np, logits, predict):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False,  #
                                          gpu_options=gpu_options)) as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=cdir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        try:
            # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            # true_count = 0  # Counts the number of correct predictions.
            # total_sample_count = num_iter * FLAGS.batch_size
            # step = 0
            # while step < num_iter and not coord.should_stop():
            #     predictions = sess.run([top_k_op])
            #     true_count += np.sum(predictions)
            #     step += 1
            # predictions = sess.run([top_k_op])
            # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size_test))
            # true_count = 0  # Counts the number of correct predictions.
            # total_sample_count = num_iter * FLAGS.batch_size_test
            # step = 0
            # while step < num_iter and not coord.should_stop():
            _, logits, pred = sess.run([top_k_op, logits, predict])

            # with open("testresultss_LSTM.txt", "a") as myfile:
            #     myfile.write(np.array2string(logits) + '\n')

            # print(pred)
            # logits_max = np.argmax(logits,axis=1)
            # print(logits_max)
            # print(labels_np)

            precision = precision_score(labels_np, pred)
            recall = recall_score(labels_np, pred)
            acc = accuracy_score(labels_np, pred)
            cm = confusion_matrix(labels_np, pred)
            # recall = sess.run(rec_op)
            # acc = sess.run(acc_op)#accuracy

            # true_count = np.sum(predictions)
            # false_count = FLAGS.num_examples - np.count_nonzero(predictions)
            # Compute precision @ 1.
            # precision = true_count / FLAGS.num_examples
            # recall = tf.metrics.recall(labels=labels,predictions=predictions)
            print('precision @ 1 = %.3f recall @ 1 = %.3f acc @ 1 = %.3f' % (precision, recall, acc))
            with open("testresultss_LSTM.txt", "a") as myfile:
                # myfile.write(cdir + '\n')
                myfile.write(cdir + ',%.3f,%.3f,%.3f \n' % (precision, recall, acc))
                myfile.write(np.array2string(cm) + '\n')

        # summary = tf.Summary()
        # summary.ParseFromString(sess.run(summary_op))
        # summary.value.add(tag='Precision @ 1', simple_value=precision)
        # summary.value.add(tag='Recall @ 1', simple_value=recall)
        # summary.value.add(tag='Accuracy @ 1', simple_value=acc)
        # summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            print('error in ' + cdir)


def evaluate(fname, data, labels_np):
    with tf.Graph().as_default() as g:
        # with tf.device('/cpu:0'):
        # Get images and labels for CIFAR-10.
        # eval_data = FLAGS.eval_data == 'test'
        images = tf.convert_to_tensor(data, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels_np, dtype=tf.int32)
        hidden_layer = FLAGS.hidden_layer2
        # Build a Graph that computes the logits predictions from the
        # inference model.
        seq_length = LSTMProteinSequenceMain.SEQUENCELENGTH  # 128 timesteps per series
        feature_size = LSTMProteinSequenceMain.SIZEEMBEDDING

        logits = LSTMComplex.inference(images, seq_length, feature_size=feature_size, n_hidden=hidden_layer, test=True)
        # logits = tf.Print(logits, [logits], "Logits are: ")
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        predictions = tf.argmax(logits, 1)
        # print(predictions)
        # print(labels)
        # rec, rec_op = tf.metrics.recall(labels=labels, predictions=predictions)
        # pre, pre_op = tf.metrics.precision(labels=labels, predictions=predictions)
        # acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            LSTMComplex.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        try:
            saver = tf.train.Saver(variables_to_restore)
        except Exception as e:  # pylint: disable=broad-except
            print('error in ' + fmainDir)
        # Build the summary operation based on the TF collection of Summaries.
        # summary_op = tf.summary.merge_all()
        #
        # summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        eval_once(fname, saver, top_k_op, labels_np, logits, predictions)


def main(argv=None):  # pylint: disable=unused-argument
    tf.logging.set_verbosity(tf.logging.WARN)
    protein_seq_dict = Utils.load_obj('protein_seq_dict_60')
    # print(len(protein_seq_dict))
    for pName in protein_seq_dict.keys():
        foldername = fmainDir + pName
        if os.path.exists(foldername):
            targeted_protein = list(protein_seq_dict.keys()).index(pName)
            labels = []
            data = []
            label = 0
            len_sequences = 0
            for family in protein_seq_dict:

                for sequences in protein_seq_dict[family]:

                    flat_list = [item for sublist in sequences for item in sublist]
                    if len(flat_list) % (
                            LSTMProteinSequenceMain.SEQUENCELENGTH * LSTMProteinSequenceMain.SIZEEMBEDDING) == 0:
                        data.append(flat_list)
                        len_sequences += len(flat_list)
                        labels.append(label)
                        # if label == CNNComplex.FLAGS.protein_no:
                        #     labels.append(1)
                        # else:
                        #     labels.append(0)
                        # data.append(np.array(flat_list,dtype=np.float16))
                        # labels.append(1 if label==protein_no else 0)
                label += 1
                # if label > 100:
                #     break
            labels = np.asarray(labels, dtype=np.int32)
            unique, counts = np.unique(labels, return_counts=True)
            counts_seq = dict(zip(unique, counts))
            print(len(labels))
            # num_seq = counts_seq[targeted_protein]
            start = 0
            for i in range(0, targeted_protein):
                if counts_seq.has_key(i):
                    start += counts_seq[i]
            for i in range(0, len(labels)):
                labels[i] = 1 if labels[i] == targeted_protein else 0
            # index = np.argmax(labels > 0)

            data = np.asarray(data, dtype=np.float32)
            data = np.reshape(data,
                              (len(labels), LSTMProteinSequenceMain.SEQUENCELENGTH,
                               LSTMProteinSequenceMain.SIZEEMBEDDING))
            print(start, len(labels))

            data = data[start - FLAGS.num_examples:start + FLAGS.num_examples]
            labels = labels[start - FLAGS.num_examples:start + FLAGS.num_examples]

            assert data.shape[0] == labels.shape[0]
            print('eval: ', pName)
            a = datetime.now()
            evaluate(foldername, data, labels)
            b = datetime.now()
            print(b - a)
            # break


if __name__ == '__main__':
    tf.app.run()
