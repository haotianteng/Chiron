# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Tue Mar 20 23:22:12 2018

"""Variable wrapper script"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def _variable_on_cpu(name, shape, initializer,dtype = tf.float32):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
     shape: list of ints
     initializer: initializer for Variable
    Returns:
     Variable Tensor
    """
    with tf.device('/cpu:0'):
     var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer,dtype = tf.float32):
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
    var = _variable_on_cpu(
        name,
        shape,
        initializer,
        dtype = dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var