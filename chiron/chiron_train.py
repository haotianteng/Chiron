# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:32:32 2017

@author: haotianteng
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import time
from distutils.dir_util import copy_tree
import tensorflow as tf
from .chiron_queue_input import inputs
from .cnn import getcnnfeature
from .cnn import getcnnlogit
from six.moves import range


def save_model(log_dir, model_name):
    """Copy the training model folder into the log.
    TODO: Need a more orgnaized way to save the Neural Network sturcture instead of copying the whole folder into log.
    Args:
        log_dir: String, the directory of the log.
        model_name: String, the model name saved.


    """
    copy_tree(os.path.dirname(os.path.abspath(__file__)),
              log_dir + model_name + '/model')


def inference(x, seq_len, training):
    """Infer a logits of the input signal batch.

    Args:
        x: Tensor of shape [batch_size, max_time], a batch of the input signal with a maximum length `max_time`.
        seq_len: Scalar float, the maximum length of the sample in the batch.
        training: Placeholder of Boolean, Ture if the inference is during training.

    Returns:
        logits: Tensor of shape [batch_size, max_time, class_num]
        ratio: Scalar float, the scale factor between the output logits and the input maximum length.
    """
    cnn_feature = getcnnfeature(x, training=training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = seq_len / feashape[1]
    logits = getcnnlogit(cnn_feature)
    return logits, ratio


def dense2sparse(label):
    """Transfer the dense label tensor to sparse tensor, the padding value should be -1 for the input dense label.

    Input:
        label
    Output:
        A tf.SparseTensor of the input tensor.
    Args:
        label: Tensor of shape [batch_size, LABEL_LEN], padded with -1.

    Returns:
        SparseTensor, the sparse format of the label.
    """
    idx = tf.where(tf.not_equal(label, -1))
    sparse = tf.SparseTensor(idx, tf.gather_nd(label, idx), label.get_shape())
    return sparse


def loss(logits, seq_len, label):
    """Calculate a CTC loss from the input logits and label.

    Args:
        logits: Tensor of shape [batch_size,max_time,class_num], logits from last layer of the NN, usually from a
            Fully connected layyer.
        seq_len: Tensor of shape [batch_size], sequence length for each sample in the batch.
        label: A Sparse Tensor of labels, sparse tensor of the true label.

    Returns:
        Tensor of shape [batch_size], losses of the batch.
    """
    loss = tf.reduce_mean(
        tf.nn.ctc_loss(label, logits, seq_len, ctc_merge_repeated=True,
                       time_major=False))
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss', loss)
    return loss


def train_step(loss, step_rate, global_step=None):
    """Generate training op

    Args:
        loss: Tensor of shape [batch_size].
        step_rate: Scalar tensor or float, the learning rate of the optimizer.
        global_step: Scalar tensor, the global step recorded.

    Returns:

    """
    opt = tf.train.AdamOptimizer(step_rate).minimize(loss,
                                                     global_step=global_step)
    #    Uncomment to use different optimizer
    #    opt = tf.train.GradientDescentOptimizer(FLAGS.step_rate).minimize(loss)
    #    opt = tf.train.RMSPropOptimizer(FLAGS.step_rate).minimize(loss)
    #    opt = tf.train.MomentumOptimizer(FLAGS.step_rate,0.9).minimize(loss)
    return opt


def prediction(logits, seq_length, label, top_paths=1):
    """Calculate the edit distance from a given a logits and label sequence.

    Args:
        logits: Tensor of shape [batch_size,max_time,class_num], input logits.
        seq_length: Tensor of shape [batch_size], sequence length of each sample in the batch.
        label: Sparse tensor.
        top_paths: Int, The number of top score path to choose from the decorder.

    Returns:
        Scalar Tensor, the mean edit distance(error rate) of the batch.
    """
    logits = tf.transpose(logits, perm=[1, 0, 2])
    predict = \
    tf.nn.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False,
                                  top_paths=top_paths)[0]
    edit_d = list()
    for i in range(top_paths):
        tmp_d = tf.edit_distance(tf.to_int32(predict[i]), label, normalize=True)
        edit_d.append(tmp_d)
    tf.stack(edit_d, axis=0)
    d_min = tf.reduce_min(edit_d, axis=0)
    error = tf.reduce_mean(d_min, axis=0)
    tf.summary.scalar('Error_rate', error)
    return error


def train(hparam):
    """Main training function.
    This will train a Neural Network with the given dataset.

    Args:
        hparam: hyper parameter for training the neural network
            TODO: Need a more detailed explanation here.
    """
    training = tf.placeholder(tf.bool)
    global_step = tf.get_variable('global_step', trainable=False, shape=(),
                                  dtype=tf.int32,
                                  initializer=tf.zeros_initializer())

    x, seq_length, train_labels = inputs(hparam.data_dir, hparam.batch_size,
                                         for_valid=False)
    y = dense2sparse(train_labels)

    logits, ratio = inference(x, hparam.sequence_len, training)
    ctc_loss = loss(logits, seq_length, y)
    opt = train_step(ctc_loss, hparam.step_rate, global_step=global_step)
    error = prediction(logits, seq_length, y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    save_model(hparam.log_dir, hparam.model_name)
    if not hparam.retrain:
        sess.run(init)
        print("Model init finished, begin training. \n")
    else:
        saver.restore(sess, tf.train.latest_checkpoint(
            hparam.log_dir + hparam.model_name))
        print("Model loaded finished, begin training. \n")
    summary_writer = tf.summary.FileWriter(
        hparam.log_dir + hparam.model_name + '/summary/', sess.graph)

    _ = tf.train.start_queue_runners(sess=sess)

    start = time.time()
    for i in range(hparam.max_steps):
        feed_dict = {training: True}
        loss_val, _ = sess.run([ctc_loss, opt], feed_dict=feed_dict)
        if i % 10 == 0:
            global_step_val = tf.train.global_step(sess, global_step)
            feed_dict = {training: True}
            error_val = sess.run(error, feed_dict=feed_dict)
            end = time.time()
            print(
                "Step %d/%d ,  loss: %5.3f edit_distance: %5.3f Elapsed Time/batch: %5.3f" \
                % (i, hparam.max_steps, loss_val, error_val,
                   (end - start) / (i + 1)))
            saver.save(sess, hparam.log_dir + hparam.model_name + '/model.ckpt',
                       global_step=global_step_val)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step_val)
            summary_writer.flush()
    global_step_val = tf.train.global_step(sess, global_step)
    print("Model %s saved." % (hparam.log_dir + hparam.model_name))
    saver.save(sess, hparam.log_dir + hparam.model_name + '/final.ckpt',
               global_step=global_step_val)


def run(hparam):
    train(hparam)


if __name__ == "__main__":
    class Flags():
        def __init__(self):
            # todo: remove hard-coded file paths
            self.data_dir = '/media/Linux_ex/Nanopore_Data/Lambda_R9.4/file_batch/'
            self.log_dir = '/media/Linux_ex/GVM_model/'
            self.sequence_len = 512
            self.batch_size = 100
            self.step_rate = 1e-3
            self.max_steps = 40000
            self.kmer = 1
            self.model_name = 'queue_res1+wavenet1x7_1x2filter'
            self.retrain = False


    flags = Flags()
    run(flags)
