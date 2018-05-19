# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Tue Mar 20 7:41:20 2018
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import argparse
import tensorflow as tf
import chiron.chiron_model as model

from six.moves import range
from chiron.chiron_queue_input import inputs
from distutils.dir_util import copy_tree
from tensorflow.contrib.training.python.training import hparam

def tower_loss(scope,x,seqlen, labels,full_seq_len,config):
    """Calculating the loss on a single GPU.
    
    Args:
        scope (String): prefix string describe the tower name, e.g. 'tower_0'
        x (Float): Tensor of shape [batch_size, max_time], batch of input signal.
        seqlen (Int): Tensor of shape [batch_size], length of sequence in batch.
        labels (Int): Sparse Tensor, true labels.
        config  (Dict): Key-Value pairs of the model configuration

    Returns:
        Tensor of shape [batch_size] containing the loss for a batch of data.
    """
    logits,_ = model.inference(x,seqlen,training = True, full_sequence_len = full_seq_len,configure = config)
    sparse_labels = dense2sparse(labels)
    _ = model.loss(logits,seqlen,sparse_labels)
    error = model.prediction(logits,seqlen,sparse_labels)
    losses = tf.get_collection('losses',scope)
    total_loss = tf.add_n(losses, name='total_loss')
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)
        tf.summary.scalar(error.op.name,error)
    return total_loss,error

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        #Iterate over towers
        grads = []
        for g, _ in grad_and_vars:
            #Iterate over variables
            expanded_g = tf.expand_dims(g, axis = 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, axis=0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

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

def train(hparams):
    """Main training function.
    This will train a Neural Network with the given dataset.

    Args:
        hparams: hyper parameter for training the neural network
            data-dir: String, the path of the data(binary batch files) directory.
            log-dir: String, the path to save the trained model.
            sequence-len: Int, length of input signal.
            batch-size: Int.
            step-rate: Float, step rate of the optimizer.
            max-steps: Int, max training steps.
            kmer: Int, size of the dna kmer.
            model-name: String, model will be saved at log-dir/model-name.
            retrain: Boolean, if True, the model will be reload from log-dir/model-name.

    """
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        training = tf.placeholder(tf.bool)
        global_step = tf.get_variable('global_step', trainable=False, shape=(),
                                    dtype=tf.int32,
                                    initializer=tf.zeros_initializer())
        
        opt = model.train_opt(hparams.step_rate,hparams.max_steps,global_step = global_step)
        x, seq_length, train_labels = inputs(hparams.data_dir, int(hparams.batch_size*hparams.ngpus),
                                            for_valid=False)
        split_y = tf.split(train_labels,hparams.ngpus,axis=0)
        split_seq_length = tf.split(seq_length,hparams.ngpus,axis=0)
        split_x = tf.split(x,hparams.ngpus,axis=0)
        tower_grads = []
        default_config = os.path.join(hparams.log_dir,hparams.model_name,'model.json')
        if hparams.retrain:
            if os.path.isfile(default_config):
                config_file = default_config
            else:
                raise ValueError("Model Json file has not been found in model log directory")
        else:
            config_file = hparams.configure
        config = model.read_config(config_file)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(hparams.ngpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('gpu_tower' ,i)) as scope:
                        loss,error = tower_loss(scope,
                                                split_x[i],
                                                split_seq_length[i],
                                                split_y[i],
                                                full_seq_len = hparams.sequence_len,
                                                config = config)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        for grad,var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients',grad))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name,var))
        apply_gradient_op = opt.apply_gradients(grads,global_step = global_step)
        var_averages = tf.train.ExponentialMovingAverage(
                                                    decay = model.MOVING_AVERAGE_DECAY)
        var_averages_op = var_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, var_averages_op)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False))
        model.save_model(default_config,config)
        if not hparams.retrain:
            sess.run(init)
            print("Model init finished, begin training. \n")
        else:
            saver.restore(sess, tf.train.latest_checkpoint(
                hparams.log_dir + hparams.model_name))
            print("Model loaded finished, begin training. \n")
        summary_writer = tf.summary.FileWriter(
            hparams.log_dir + hparams.model_name + '/summary/', sess.graph)
        _ = tf.train.start_queue_runners(sess=sess)

        start = time.time()
        for i in range(hparams.max_steps):
            feed_dict = {training: True}
            loss_val, _ = sess.run([loss,train_op], feed_dict=feed_dict)
            if i % 10 == 0:
                global_step_val = tf.train.global_step(sess, global_step)
                feed_dict = {training: True}
                error_val = sess.run(error, feed_dict=feed_dict)
                end = time.time()
                print(
                    "Step %d/%d ,  loss: %5.3f edit_distance: %5.3f Elapsed Time/batch: %5.3f" \
                    % (i, hparams.max_steps, loss_val, error_val,
                    (end - start) / (i + 1)))
                saver.save(sess, hparams.log_dir + hparams.model_name + '/model.ckpt',
                        global_step=global_step_val)
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=global_step_val)
                summary_writer.flush()
        global_step_val = tf.train.global_step(sess, global_step)
        print("Model %s saved." % (hparams.log_dir + hparams.model_name))
        saver.save(sess, hparams.log_dir + hparams.model_name + '/final.ckpt',
                global_step=global_step_val)


def run(hparams):
    train(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '-i',
            '--data-dir',
            help='Location containing binary training data',
            default = '/media/Linux_ex/Nanopore_Data/20170322_c4_watermanag_S10/file_batch')
#            required=True)
   
    parser.add_argument(
            '-o',
            '--log-dir',
            help='Log dir location',
            default = '/media/Linux_ex/model_test')
#            required=True)
    parser.add_argument(
            '-m',
            '--model-name',
            help='Model name',
            default = "test")
#            required=True)
    parser.add_argument(
           '--configure',
           default = None,
           help="Model structure configure json file.")
    parser.add_argument(
            '-n',
            '--ngpus',
            help='number of gpus',
            default = 1,
            type = int)
    parser.add_argument(
            '-s',
            '--sequence-len',
            help='Sequence length of nucleotides',
            default=512,
            type=int)
    parser.add_argument(
            '-b',
            '--batch-size',
            help='Training batch size',
            default=200,
            type=int)
    parser.add_argument(
            '-t',
            '--step-rate',
            help='Step rate',
            default=1e-3,
            type=float)
    parser.add_argument(
            '-x',
            '--max-steps',
            help='Max training steps',
            default=20000,
            type=int)
    parser.add_argument(
            '-k'
            '--kmer',
            help='K-mer length',
            default=1,
            type=int)
    parser.add_argument(
            '-r',
            '--retrain',
            help='Flag if retrain the model',
            default=False,
            type=bool)
    args = parser.parse_args()
    run(hparam.HParams(**args.__dict__)) 
