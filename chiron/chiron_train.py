# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Mon Apr 17 17:32:32 2017

from __future__ import absolute_import
from __future__ import print_function
import os
import time
import argparse
import tensorflow as tf
import chiron.chiron_model as model

from six.moves import range
from chiron.chiron_queue_input import inputs
from distutils.dir_util import copy_tree
from tensorflow.contrib.training.python.training import hparam

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

def train(hparam):
    """Main training function.
    This will train a Neural Network with the given dataset.

    Args:
        hparam: hyper parameter for training the neural network
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
    training = tf.placeholder(tf.bool)
    global_step = tf.get_variable('global_step', trainable=False, shape=(),
                                  dtype=tf.int32,
                                  initializer=tf.zeros_initializer())

    x, seq_length, train_labels = inputs(hparam.data_dir, hparam.batch_size,
                                         for_valid=False)
    y = dense2sparse(train_labels)
    default_config = os.path.join(hparam.log_dir,hparam.model_name,'model.json')
    if hparam.retrain:
        if os.path.isfile(default_config):
            config_file = default_config
        else:
            raise ValueError("Model Json file has not been found in model log directory")
    else:
        config_file = hparam.configure
    config = model.read_config(config_file)
    logits, _ = model.inference(x,seq_length,training,hparam.sequence_len,configure = config)
    ctc_loss = model.loss(logits, seq_length, y)
    opt = model.train_opt(hparam.step_rate,hparam.max_steps,global_step = global_step)
    step = opt.minimize(ctc_loss,global_step = global_step)
    error = model.prediction(logits, seq_length, y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model.save_model(default_config, config)
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
        loss_val, _ = sess.run([ctc_loss, step], feed_dict=feed_dict)
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
   parser = argparse.ArgumentParser()

   parser.add_argument(
           '-i',
           '--data-dir',
           help='Location containing binary training data',
           required=True)
   
   parser.add_argument(
           '-o',
           '--log-dir',
           help='Log dir location',
           required=True)
   parser.add_argument(
           '-m',
           '--model-name',
           help='Model name',
           required=True)
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
           '--configure',
           default = None,
           help="Model structure configure json file.")
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
