# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import CNNinput

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_boolean('save_summaries', True,
                            """Should save summaries or not""")

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
# IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = CNNinput.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = CNNinput.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 300.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _generate_image_and_label_batch2(image, label, batch_size=FLAGS.batch_size, shuffle=True):
    min_fraction_of_examples_in_queue = 0.4
    min_after_dequeue = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                            min_fraction_of_examples_in_queue)
    dataset = tf.data.Dataset.from_tensor_slices((image, label))
    # .map(lambda x,y:_parse_and_preprocess(x,y))\
    dataset = dataset.shuffle(min_after_dequeue).repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    # images, labels =
    return iterator.get_next()


def _generate_image_and_label_batch(image, label,
                                    batch_size=FLAGS.batch_size, shuffle=True):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d data before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    num_preprocess_threads = 4

    # dataset = tf.data.Dataset.from_tensor_slices(((image), label))

    # Shuffle, repeat, and batch the examples.

    if shuffle:
        # dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        images, label_batch = tf.train.shuffle_batch(
            [image[0], label[0]],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.summary.image('images', images)
    # tf.Print(label_batch,[label_batch],message="label = ")
    #
    return images, tf.reshape(label_batch, [batch_size])


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(_X, seq_length, feature_size, n_hidden,test=False):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    # _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    # shape = _X.get_shape().as_list()
    # Move everything into depth so we can perform a single matrix multiply..
    # _X = tf.reshape(_X, [-1, shape[1] * shape[2] * shape[3]])

    _weights = {
        'hidden': _variable_with_weight_decay('weights_hidden', shape=[feature_size, n_hidden], stddev=1, wd=None),#tf.Variable(tf.random_normal([n_input, hidden_layer])),  # Hidden layer weights
        'out':  _variable_with_weight_decay('weights_out',shape=[n_hidden, NUM_CLASSES],stddev=1,wd=None)#tf.Variable(tf.random_normal([n_hidden, NUM_CLASSES], mean=1.0))
    }
    _biases = {
        'hidden': _variable_with_weight_decay('bias_hidden',shape=[n_hidden],stddev=0,wd=None),#tf.Variable(tf.random_normal([n_hidden])),
        'out': _variable_with_weight_decay('bias_out',shape=[NUM_CLASSES],stddev=0,wd=None)
    }
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, feature_size])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    _activation_summary(_X)
    if not test:
        _X = tf.nn.dropout(_X, keep_prob=.8)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, seq_length, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    if not test:
        lstm_cell_1 =tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1,output_keep_prob=.7)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    if not test:
        lstm_cell_2= tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, output_keep_prob=.8)

    lstm_cell_3 = tf.nn.rnn_cell.GRUCell(n_hidden)
    # lstm_cell_3 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_3, output_keep_prob=1)

    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2,lstm_cell_3], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


# def inference(images):
#     """Build the CIFAR-10 model.
#
#     Args:
#       images: Images returned from distorted_inputs() or inputs().
#
#     Returns:
#       Logits.
#     """
#     # We instantiate all variables using tf.get_variable() instead of
#     # tf.Variable() in order to share variables across multiple GPU training runs.
#     # If we only ran this model on a single GPU, we could simplify this function
#     # by replacing all instances of tf.get_variable() with tf.Variable().
#     #
#     # conv1
#     with tf.variable_scope('conv1') as scope:
#         kernel = _variable_with_weight_decay('weights',
#                                              shape=[3, 3, 1, 64],
#                                              stddev=5e-2,
#                                              wd=None)
#         conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv1 = tf.nn.relu(pre_activation, name=scope.name)
#         _activation_summary(conv1)
#
#     # pool1
#     pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                            padding='SAME', name='pool1')
#     # norm1
#     norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm1')
#
#     # conv2
#     with tf.variable_scope('conv2') as scope:
#         kernel = _variable_with_weight_decay('weights',
#                                              shape=[3, 3, 64, 64],
#                                              stddev=5e-2,
#                                              wd=None)
#         conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2 = tf.nn.relu(pre_activation, name=scope.name)
#         # _activation_summary(conv2)
#
#     # norm2
#     norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm2')
#     # pool2
#     pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
#                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')
#
#     # local3
#     with tf.variable_scope('local3') as scope:
#         shape = pool2.get_shape().as_list()
#         # Move everything into depth so we can perform a single matrix multiply..
#         reshape = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
#         dim = reshape.get_shape()[1].value
#         # reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
#         # dim = reshape.get_shape()[1].value
#         weights = _variable_with_weight_decay('weights', shape=[dim, 384],
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#         # _activation_summary(local3)
#
#     # local4
#     with tf.variable_scope('local4') as scope:
#         weights = _variable_with_weight_decay('weights', shape=[384, 192],
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
#         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
#         # _activation_summary(local4)
#
#     # linear layer(WX + b),
#     # We don't apply softmax here because
#     # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
#     # and performs the softmax internally for efficiency.
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
#                                               stddev=1 / 192.0, wd=None)
#         biases = _variable_on_cpu('biases', [NUM_CLASSES],
#                                   tf.constant_initializer(0.0))
#         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
#         # _activation_summary(softmax_linear)
#
#     return softmax_linear


def loss(logits, labels, lambda_loss_amount=0.002):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=labels, logits=logits, name='cross_entropy_per_example')
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # tf.add_to_collection('losses', cross_entropy_mean)

    # l2 = lambda_loss_amount * sum(
    #     tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    # )  # L2 loss prevents this overkill neural network to overfit the data
    labels = tf.cast(labels, tf.int64)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #     labels=tf.one_hot(labels,depth=NUM_CLASSES), logits=logits, name='cross_entropy_per_example')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('cross_entropy_losses', cross_entropy_mean)
    # tf.add_to_collection('l2_losses', l2)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('cross_entropy_losses'), name='total_loss_cross_l2')

    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    # NUM_CLASSES = num_classes

    """Train LSTM model

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    if FLAGS.save_summaries:
        tf.summary.scalar('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.

    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(learning_rate=lr)  # .minimize(total_loss)

        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if FLAGS.save_summaries:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None and tf.is_nan(grad) is None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op
