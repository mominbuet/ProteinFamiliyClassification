from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import datetime
import Utils
import numpy as np
import tensorflow as tf
import CNNProteinSequenceMain
import CNNComplex
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix


ProteinName = 'PF01288'
fmainDir ='./home/azizmma/HIPPIStore/tmp'
FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('eval_dir', './store/tmp' + ProteinName + '/eval',
#                            """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', fmainDir+ ProteinName + '/',
#                            """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")


def eval_once(cdir,saver,  top_k_op,  labels_np, logits, predict):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    try:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        with tf.Session(config=tf.ConfigProto( log_device_placement=False,#
                                              gpu_options=gpu_options)) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=cdir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            predictions, logits, pred = sess.run([top_k_op, logits, predict])
            # print(logits)
            # logits_max = np.argmax(logits,axis=1)
            # print(logits_max)
            # print(labels_np)

            precision = precision_score(labels_np, pred)
            recall = recall_score(labels_np, pred)
            acc = accuracy_score(labels_np, pred)
            cm = confusion_matrix(labels_np, pred)
            # print(cm)
            # recall = sess.run(rec_op)
            # acc = sess.run(acc_op)#accuracy

            # true_count = np.sum(predictions)
            # false_count = FLAGS.num_examples - np.count_nonzero(predictions)
            # Compute precision @ 1.
            # precision = true_count / FLAGS.num_examples
            # recall = tf.metrics.recall(labels=labels,predictions=predictions)
            print('precision @ 1 = %.3f recall @ 1 = %.3f acc @ 1 = %.3f' % ( precision, recall, acc))
            with open("testresultss_danta8_2.txt", "a") as myfile:
                myfile.write(cdir+'\n')
                myfile.write('precision @ 1 = %.3f recall @ 1 = %.3f acc @ 1 = %.3f \n' % ( precision, recall, acc))
                myfile.write(np.array2string(cm)+'\n')
                # myfile.write(np.array2string(logits) + '\n')
            # summary = tf.Summary()
            # summary.ParseFromString(sess.run(summary_op))
            # summary.value.add(tag='Precision @ 1', simple_value=precision)
            # summary.value.add(tag='Recall @ 1', simple_value=recall)
            # summary.value.add(tag='Accuracy @ 1', simple_value=acc)
            # summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        print('Error for ',cdir)
    #     coord.request_stop(e)
    #
    # coord.request_stop()
    # coord.join(threads, stop_grace_period_secs=10)


def evaluate(cdir, data, labels_np):
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            # Get images and labels for CIFAR-10.
            eval_data = FLAGS.eval_data == 'test'
            images = tf.convert_to_tensor(data, dtype=tf.float32)
            labels = tf.convert_to_tensor(labels_np, dtype=tf.int32)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = CNNComplex.inference(images)
            # logits = tf.Print(logits, [logits], "Logits are: ")
            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            predictions = tf.argmax(logits, 1)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                CNNComplex.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            # summary_op = tf.summary.merge_all()

            # summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

            # while True:
            eval_once(cdir,saver, top_k_op,  labels_np, logits, predictions)


def main(argv=None):  # pylint: disable=unused-argument
    tf.logging.set_verbosity(tf.logging.WARN)
    protein_seq_dict = Utils.load_obj('protein_seq_dict')
    # print(len(protein_seq_dict))
    for pName in protein_seq_dict.keys():
        foldername = fmainDir+pName
        if os.path.exists(foldername):
            targeted_protein =list(protein_seq_dict.keys()).index(pName)
            labels = []
            data = []
            label = 0
            len_sequences = 0
            for family in protein_seq_dict:

                for sequences in protein_seq_dict[family]:

                    flat_list = [item for sublist in sequences for item in sublist]
                    if len(flat_list) % (CNNProteinSequenceMain.SEQUENCELENGTH * CNNProteinSequenceMain.SIZEEMBEDDING) == 0:
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
                              (len(labels), CNNProteinSequenceMain.SEQUENCELENGTH, CNNProteinSequenceMain.SIZEEMBEDDING, 1))
            print(start,  len(labels))

            data = data[start-FLAGS.num_examples:start+FLAGS.num_examples]
            labels = labels[start-FLAGS.num_examples:start+FLAGS.num_examples]


            assert data.shape[0] == labels.shape[0]
            print('eval: ',pName)
            evaluate(foldername,data, labels)
            break
            # b = datetime.datetime.now()

if __name__ == '__main__':

    tf.app.run()
