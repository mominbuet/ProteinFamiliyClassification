from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time       
import numpy as np
import tensorflow as tf

import CNNComplex
import  CNNProteinSequenceMain
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
# tf.app.flags.DEFINE_integer('save_checkpoint', 50000,
#                             """How often to log results to the console.""")

def train(data, labels, NUM_CLASSES,data_must,label_must,train_dir):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            # data = tf.convert_to_tensor(data, dtype=tf.float32)
            # labels_tf = tf.transpose(tf.convert_to_tensor(labels, dtype=tf.int32))
            # dataset = tf.data.Dataset.from_tensor_slices(((data), labels_tf))
            # labels_tf = tf.one_hot(labels_tf,len(labels))
            # labels_tf = labels_tf.set_shape([1])
            # dataset = dataset.shuffle(1000).repeat().batch(FLAGS.batch_size)
            # iterator = dataset.make_one_shot_iterator()
            # images, labels = iterator.get_next()
            # np.random.seed(100)

            # images = tf.convert_to_tensor(data, np.float32)
            # labels = tf.convert_to_tensor(labels, np.int32)

            images, labels = CNNComplex._generate_image_and_label_batch2(data,labels)
            images = tf.concat([data_must,images],0)
            labels = tf.concat([label_must, labels ], 0)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # data = tf.reshape(data,shape=(tf.shape(labels)[0],60,8,1))
        logits = CNNComplex.inference(images)

        # labels= tf.Print(labels,[labels],"labels:")
        # Calculate loss.
        loss = CNNComplex.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = CNNComplex.train(loss, global_step, NUM_CLASSES)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * (FLAGS.batch_size-FLAGS.must_contain) / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # config.gpu_options.allow_growth = True

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_dir,
                # save_checkpoint_steps=FLAGS.save_checkpoint,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=config) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

#
# def main(argv=None):  # pylint: disable=unused-argument
#     # CNNComplex.maybe_download_and_extract()
#     # if tf.gfile.Exists(FLAGS.train_dir):
#     #     tf.gfile.DeleteRecursively(FLAGS.train_dir)
#     # tf.gfile.MakeDirs(FLAGS.train_dir)
#     train()
#
#
# if __name__ == '__main__':
#     tf.app.run()
