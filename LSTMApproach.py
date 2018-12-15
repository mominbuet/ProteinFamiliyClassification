import numpy as np

import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import Utils
import os
# import CNNComplex
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('save_checkpoint', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('must_contain', 30,
                            """must contain how many from beginning""")
tf.app.flags.DEFINE_integer('train_perc', 80,
                            """Training percentage""")
SIZEEMBEDDING = 8
SEQUENCELENGTH = 100




def main(unused_argv):
    targeted_protein_start =1# int(unused_argv[1])
    # Load training and eval data
    protein_seq_dict = Utils.load_obj('protein_seq_dict')
    labels_original = []
    data = []
    label = 0
    len_sequences = 0
    for family in protein_seq_dict:
        for sequences in protein_seq_dict[family]:

            flat_list = [item for sublist in sequences for item in sublist]
            if len(flat_list) % (SEQUENCELENGTH * SIZEEMBEDDING) == 0:
                data.append(flat_list)
                len_sequences += len(flat_list)
                labels_original.append(label)
        label += 1
    labels_original = np.asarray(labels_original, dtype=np.int32)
    # print(len(protein_seq_dict))
    for start_from in range(targeted_protein_start, 2298):
        print("currently considering " + list(protein_seq_dict.keys())[start_from])
        labels = np.zeros(len(labels_original))
        unique, counts = np.unique(labels_original, return_counts=True)
        counts_seq = dict(zip(unique, counts))
        num_seq = 0
        if counts_seq.has_key(start_from):
            num_seq = counts_seq[start_from]
        if num_seq < 10:
            print("Too few (<10) sequences for " + list(protein_seq_dict.keys())[start_from])
            continue
        start = 0
        for i in range(0, start_from):
            if counts_seq.has_key(i):
                start += counts_seq[i]
        for i in range(0, len(labels_original)):
            labels[i] = 1 if labels_original[i] == start_from else 0
    # index = np.argmax(labels > 0)
        train_dir = './LSTM' + list(protein_seq_dict.keys())[start_from] + '/'

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

        split = int((FLAGS.train_perc/100)* float(len(labels)))-len(label_must)

        X_train = data[:split, ]
        Y_train = labels[:split]
        np.concatenate((X_train,data_must[:len(label_must)-10]),axis=0)
        np.concatenate((Y_train, label_must[:len(label_must)-10]),axis=0)

        X_test = data[split:, ]
        y_test = labels[split:]
        np.concatenate((X_test, data_must[len(label_must)-10:]),axis=0)
        np.concatenate((y_test, label_must[len(label_must)-10:]),axis=0)

        training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
        test_data_count = len(X_test)  # 2947 testing series
        n_steps = SEQUENCELENGTH# 128 timesteps per series
        n_input =   SIZEEMBEDDING  # 9 input parameters per timestep

        # LSTM Neural Network's internal structure

        n_hidden = 128  # Hidden layer num of features
        n_classes = 2  # Total classes (should go up, or should go down)

        # Training

        learning_rate = 0.0025
        lambda_loss_amount = 0.0015
        training_iters = FLAGS.max_steps # Loop 300 times on the dataset
        batch_size = FLAGS.batch_size
        display_iter = 10000  # To show test set accuracy during training

        # Some debugging info

        print("Some useful info to get an insight on dataset's shape and normalisation:")
        print("(X shape, y shape, every X's mean, every X's standard deviation)")
        print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
        print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

        def extract_batch_size(_train, step, batch_size):
            # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

            shape = list(_train.shape)
            shape[0] = batch_size
            batch_s = np.empty(shape)

            for i in range(batch_size):
                # Loop index
                index = ((step - 1) * batch_size + i) % len(_train)
                batch_s[i] = _train[index]

            return batch_s

        def LSTM_RNN(_X, _weights, _biases):
            # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
            # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
            # Note, some code of this notebook is inspired from an slightly different
            # RNN architecture used on another dataset, some of the credits goes to
            # "aymericdamien" under the MIT license.

            # (NOTE: This step could be greatly optimised by shaping the dataset once
            # input shape: (batch_size, n_steps, n_input)
            _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
            # Reshape to prepare input to hidden activation
            _X = tf.reshape(_X, [-1, n_input])
            # new shape: (n_steps*batch_size, n_input)

            # ReLU activation, thanks to Yu Zhao for adding this improvement here:
            _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            _X = tf.split(_X, n_steps, 0)
            # new shape: n_steps * (batch_size, n_hidden)

            # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
            lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
            # Get LSTM cell output
            outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

            # Get last time step's output feature for a "many-to-one" style classifier,
            # as in the image describing RNNs at the top of this page
            lstm_last_output = outputs[-1]

            # Linear activation
            return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

        def one_hot(y_, n_classes=n_classes):
            # Function to encode neural one-hot output labels from number indexes
            # e.g.:
            # one_hot(y_=[[5], [0], [3]], n_classes=6):
            #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

            y_ = y_.reshape(len(y_))
            return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


        # Graph input / output
        x = tf.placeholder(tf.float32, [None, n_steps, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])

        # Graph weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        pred = LSTM_RNN(x, weights, biases)

        # Loss, optimizer and evaluation
        l2 = lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        )  # L2 loss prevents this overkill neural network to overfit the data
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2  # Softmax loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []

        # Launch the graph
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                device_count={'GPU': 0})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        # sess = tf.train.MonitoredTrainingSession(
        #     checkpoint_dir=train_dir,
        #     save_checkpoint_steps=FLAGS.save_checkpoint,
        #     hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
        #            # tf.train.NanTensorHook(loss),
        #            # _LoggerHook()
        #            ],
        #     config=config)

        saver = tf.train.Saver()

        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        # Perform Training steps with "batch_size" amount of example data at each loop
        step = 1
        while step * batch_size <= training_iters:
            batch_xs = extract_batch_size(X_train, step, batch_size)
            batch_ys = one_hot(extract_batch_size(Y_train, step, batch_size))
            # batch_xs,batch_ys = CNNComplex._generate_image_and_label_batch2(X_train,Y_train)
            # Fit training using batch data
            _, loss, acc = sess.run(
                [optimizer, cost, accuracy],
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            )
            train_losses.append(loss)
            train_accuracies.append(acc)

            # Evaluate network only at some steps for faster training:
            if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
                # To not spam console, show training accuracy/loss in this "if"
                print("Training iter #" + str(step * batch_size) + \
                      ":   Batch Loss = " + "{:.6f}".format(loss) + \
                      ", Accuracy = {}".format(acc))

                # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
                loss, acc = sess.run(
                    [cost, accuracy],
                    feed_dict={
                        x: X_test,
                        y: one_hot(y_test)
                    }
                )
                test_losses.append(loss)
                test_accuracies.append(acc)

                print("PERFORMANCE ON TEST SET: " + \
                      "Batch Loss = {}".format(loss) + \
                      ", Accuracy = {}".format(acc))
            # if (step % FLAGS.max_steps == 0) and step != 0:


            step += 1

        print("Optimization Finished!")

        # Accuracy for test data

        one_hot_predictions, accuracy, final_loss = sess.run(
            [pred, accuracy, cost],
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )

        test_losses.append(final_loss)
        test_accuracies.append(accuracy)

        print("FINAL RESULT: " + \
              "Batch Loss = {}".format(final_loss) + \
              ", Accuracy = {}".format(accuracy))
        save_path = saver.save(sess, train_dir)
        print("Model saved in path: %s" % save_path)
        sess=None

if __name__ == "__main__":
    tf.app.run()
