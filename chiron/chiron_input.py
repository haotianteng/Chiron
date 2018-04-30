# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Mon Mar 27 14:04:57 2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import sys
import tempfile

import h5py
import numpy as np
from statsmodels import robust
from six.moves import range
from six.moves import zip
import tensorflow as tf

raw_labels = collections.namedtuple('raw_labels', ['start', 'length', 'base'])


class Flags(object):
    def __init__(self):
        self.max_segments_number = None
        self.MAXLEN = 1e4  # Maximum Length of the holder in biglist. 1e5 by default


#        self.max_segment_len = 200
FLAGS = Flags()


class biglist(object):
    """
    biglist class, read into memory if reads number < MAXLEN, otherwise read into a hdf5 file.
    """

    def __init__(self, 
                 data_handle, 
				 dtype='float32', 
				 length=0, 
				 cache=False,
                 max_len=1e5):
        self.handle = data_handle
        self.dtype = dtype
        self.holder = list()
        self.length = length
        self.max_len = max_len
        self.cache = cache  # Mark if the list has been saved into hdf5 or not

    @property
    def shape(self):
        return self.handle.shape

    def append(self, item):
        self.holder.append(item)
        self.check_save()

    def __add__(self, add_list):
        self.holder += add_list
        self.check_save()
        return self

    def __len__(self):
        return self.length + len(self.holder)

    def resize(self, size, axis=0):
        self.save_rest()
        if self.cache:
            self.handle.resize(size, axis=axis)
            self.length = len(self.handle)
        else:
            self.holder = self.holder[:size]

    def save_rest(self):
        if self.cache:
            if len(self.holder) != 0:
                self.save()

    def check_save(self):
        if len(self.holder) > self.max_len:
            self.save()
            self.cache = True

    def save(self):
        if type(self.holder[0]) is list:
            max_sub_len = max([len(sub_a) for sub_a in self.holder])
            shape = self.handle.shape
            for item in self.holder:
                item.extend([0] * (max(shape[1], max_sub_len) - len(item)))
            if max_sub_len > shape[1]:
                self.handle.resize(max_sub_len, axis=1)
            self.handle.resize(self.length + len(self.holder), axis=0)
            self.handle[self.length:] = self.holder
            self.length += len(self.holder)
            del self.holder[:]
            self.holder = list()
        else:
            self.handle.resize(self.length + len(self.holder), axis=0)
            self.handle[self.length:] = self.holder
            self.length += len(self.holder)
            del self.holder[:]
            self.holder = list()

    def __getitem__(self, val):
        if self.cache:
            if len(self.holder) != 0:
                self.save()
            return self.handle[val]
        else:
            return self.holder[val]


class DataSet(object):
    def __init__(self,
                 event,
                 event_length,
                 label,
                 label_length,
                 for_eval=False,
                 ):
        """Custruct a DataSet."""
        if for_eval == False:
            assert len(event) == len(label) and len(event_length) == len(
                label_length) and len(event) == len(
                event_length), "Sequence length for event \
            and label does not of event and label should be same, \
            event:%d , label:%d" % (len(event), len(label))
        self._event = event
        self._event_length = event_length
        self._label = label
        self._label_length = label_length
        self._reads_n = len(event)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._for_eval = for_eval
        self._perm = np.arange(self._reads_n)

    @property
    def event(self):
        return self._event

    @property
    def label(self):
        return self._label

    @property
    def event_length(self):
        return self._event_length

    @property
    def label_length(self):
        return self._label_length

    @property
    def reads_n(self):
        return self._reads_n

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def for_eval(self):
        return self._for_eval

    @property
    def perm(self):
        return self._perm

    def read_into_memory(self, index):
        event = np.asarray(list(zip([self._event[i] for i in index],
                                    [self._event_length[i] for i in index])))
        if not self.for_eval:
            label = np.asarray(list(zip([self._label[i] for i in index],
                                        [self._label_length[i] for i in index])))
        else:
            label = []
        return event, label

    def next_batch(self, batch_size, shuffle=True, sig_norm=False):
        """Return next batch in batch_size from the data set.
            Input Args:
                batch_size:A scalar indicate the batch size.
                shuffle: boolean, indicate if the data should be shuffled after each epoch.
            Output Args:
                inputX,sequence_length,label_batch: tuple of (indx,vals,shape)
        """
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            if shuffle:
                np.random.shuffle(self._perm)
        # Go to the next epoch
        if start + batch_size > self.reads_n:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest samples in this epoch
            rest_reads_n = self.reads_n - start
            event_rest_part, label_rest_part = self.read_into_memory(
                self._perm[start:self._reads_n])

            # Shuffle the data
            if shuffle:
                np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_reads_n
            end = self._index_in_epoch
            event_new_part, label_new_part = self.read_into_memory(
                self._perm[start:end])
            if event_rest_part.shape[0] == 0:
                event_batch = event_new_part
                label_batch = label_new_part
            elif event_new_part.shape[0] == 0:
                event_batch = event_rest_part
                label_batch = label_rest_part
            else:
                event_batch = np.concatenate((event_rest_part, event_new_part), axis=0)
                label_batch = np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            event_batch, label_batch = self.read_into_memory(
                self._perm[start:end])
        if not self._for_eval:
            label_batch = batch2sparse(label_batch)
        seq_length = event_batch[:, 1].astype(np.int32)
        return np.vstack(event_batch[:, 0]).astype(
            np.float32), seq_length, label_batch


def read_data_for_eval(file_path, 
					   start_index=0,
                       step=20, 
	                   seg_length=200, 
                       sig_norm=True,
                       reverse = False):
    """
    Input Args:
        file_path: file path to a signal file.
        start_index: the index of the signal start to read.
        step: sliding step size.
        seg_length: length of segments.
        sig_norm: if the signal need to be normalized.
        reverse: if the signal need to be reversed.
    """
    if not file_path.endswith('.signal'):
        raise ValueError('A .signal file is required.')
    else:
        event = list()
        event_len = list()
        label = list()
        label_len = list()
        f_signal = read_signal(file_path, normalize=sig_norm)
        if reverse:
            f_signal = f_signal[::-1]
        f_signal = f_signal[start_index:]
        sig_len = len(f_signal)
        for indx in range(0, sig_len, step):
            segment_sig = f_signal[indx:indx + seg_length]
            segment_len = len(segment_sig)
            padding(segment_sig, seg_length)
            event.append(segment_sig)
            event_len.append(segment_len)
        evaluation = DataSet(event=event, 
							 event_length=event_len, 
							 label=label,
                             label_length=label_len, 
							 for_eval=True)
    return evaluation


def read_cache_dataset(h5py_file_path):
    """Notice: Return a data reader for a h5py_file, call this function multiple
    time for parallel reading, this will give you N dependent dataset reader,
    each reader read independently from the h5py file."""
    hdf5_record = h5py.File(h5py_file_path, "r")
    event_h = hdf5_record['event/record']
    event_length_h = hdf5_record['event/length']
    label_h = hdf5_record['label/record']
    label_length_h = hdf5_record['label/length']
    event_len = len(event_h)
    label_len = len(label_h)
    assert len(event_h) == len(event_length_h)
    assert len(label_h) == len(label_length_h)
    event = biglist(data_handle=event_h, length=event_len, cache=True)
    event_length = biglist(data_handle=event_length_h, length=event_len,
                           cache=True)
    label = biglist(data_handle=label_h, length=label_len, cache=True)
    label_length = biglist(data_handle=label_length_h, length=label_len,
                           cache=True)
    return DataSet(event=event, event_length=event_length, label=label,
                   label_length=label_length)


def read_tfrecord(data_dir, 
                  tfrecord, 
                  h5py_file_path=None, 
                  seq_length=300, 
                  k_mer=1, 
                  max_segments_num=None):
    ###Read from raw data
    if max_segments_num is None:
        max_segments_num = FLAGS.max_segments_number
    if h5py_file_path is None:
        h5py_file_path = tempfile.mkdtemp() + '/temp_record.hdf5'
    else:
        try:
            os.remove(os.path.abspath(h5py_file_path))
        except:
            pass
        if not os.path.isdir(os.path.dirname(os.path.abspath(h5py_file_path))):
            os.mkdir(os.path.dirname(os.path.abspath(h5py_file_path)))
    with h5py.File(h5py_file_path, "a") as hdf5_record:
        event_h = hdf5_record.create_dataset('event/record', dtype='float32', shape=(0, seq_length),
                                             maxshape=(None, seq_length))
        event_length_h = hdf5_record.create_dataset('event/length', dtype='int32', shape=(0,), maxshape=(None,),
                                                    chunks=True)
        label_h = hdf5_record.create_dataset('label/record', dtype='int32', shape=(0, 0), maxshape=(None, seq_length))
        label_length_h = hdf5_record.create_dataset('label/length', dtype='int32', shape=(0,), maxshape=(None,))
        event = biglist(data_handle=event_h, max_len=FLAGS.MAXLEN)
        event_length = biglist(data_handle=event_length_h, max_len=FLAGS.MAXLEN)
        label = biglist(data_handle=label_h, max_len=FLAGS.MAXLEN)
        label_length = biglist(data_handle=label_length_h, max_len=FLAGS.MAXLEN)
        count = 0
        file_count = 0

        tfrecords_filename = data_dir + tfrecord
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

        for string_record in record_iterator:
            
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            raw_data_string = (example.features.feature['raw_data']
                                          .bytes_list
                                          .value[0])
            
            features_string = (example.features.feature['features']
                                        .bytes_list
                                        .value[0])
            fn_string = (example.features.feature['fname'].bytes_list.value[0])

            raw_data = np.fromstring(raw_data_string, dtype=np.int16)
            
            features_data = np.fromstring(features_string, dtype='S8')

            # grouping the whole array into sub-array with size = 3
            group_size = 3
            features_data = [features_data[n:n+group_size] for n in range(0, len(features_data), group_size)]

            f_signal = read_signal_tfrecord(raw_data)

            if len(f_signal) == 0:
                continue
            #try:
            f_label = read_label_tfrecord(features_data, skip_start=10, window_n=(k_mer - 1) / 2)
            #except:
            #    sys.stdout.write("Read the label fail.Skipped.")
            #    continue
            tmp_event, tmp_event_length, tmp_label, tmp_label_length = read_raw(f_signal, f_label, seq_length)
            event += tmp_event
            event_length += tmp_event_length
            label += tmp_label
            label_length += tmp_label_length
            del tmp_event
            del tmp_event_length
            del tmp_label
            del tmp_label_length
            count = len(event)
            if file_count % 10 == 0:
                if max_segments_num is not None:
                    sys.stdout.write("%d/%d events read.   \n" % (count, max_segments_num))
                    if len(event) > max_segments_num:
                        event.resize(max_segments_num)
                        label.resize(max_segments_num)
                        event_length.resize(max_segments_num)

                        label_length.resize(max_segments_num)
                        break
                else:
                    sys.stdout.write("%d lines read.   \n" % (count))
            file_count += 1

        if event.cache:
            train = read_cache_dataset(h5py_file_path)
        else:
            train = DataSet(event=event, event_length=event_length, label=label, label_length=label_length)
        return train
            
def read_raw_data_sets(data_dir, h5py_file_path=None, seq_length=300, k_mer=1, max_segments_num=FLAGS.max_segments_number):
    ###Read from raw data
    if h5py_file_path is None:
        h5py_file_path = tempfile.mkdtemp() + '/temp_record.hdf5'
    else:
        try:
            os.remove(os.path.abspath(h5py_file_path))
        except:
            pass
        if not os.path.isdir(os.path.dirname(os.path.abspath(h5py_file_path))):
            os.mkdir(os.path.dirname(os.path.abspath(h5py_file_path)))
    with h5py.File(h5py_file_path, "a") as hdf5_record:
        event_h = hdf5_record.create_dataset('event/record', dtype='float32', shape=(0, seq_length),
                                             maxshape=(None, seq_length))
        event_length_h = hdf5_record.create_dataset('event/length', dtype='int32', shape=(0,), maxshape=(None,),
                                                    chunks=True)
        label_h = hdf5_record.create_dataset('label/record', dtype='int32',
                                             shape=(0, 0),
                                             maxshape=(None, seq_length))
        label_length_h = hdf5_record.create_dataset('label/length',
                                                    dtype='int32', shape=(0,),
                                                    maxshape=(None,))
        event = biglist(data_handle=event_h, max_len=FLAGS.MAXLEN)
        event_length = biglist(data_handle=event_length_h, max_len=FLAGS.MAXLEN)
        label = biglist(data_handle=label_h, max_len=FLAGS.MAXLEN)
        label_length = biglist(data_handle=label_length_h, max_len=FLAGS.MAXLEN)
        count = 0
        file_count = 0
        for name in os.listdir(data_dir):
            if name.endswith(".signal"):
                file_pre = os.path.splitext(name)[0]
                f_signal = read_signal(data_dir + name)

                if len(f_signal) == 0:
                    continue
                try:
                    f_label = read_label(data_dir + file_pre + '.label',
                                         skip_start=10,
                                         window_n=int((k_mer - 1) / 2))
                except:
                    sys.stdout.write("Read the label %s fail.Skipped." % (name))
                    continue

                tmp_event, tmp_event_length, tmp_label, tmp_label_length = \
                    read_raw(f_signal, f_label, seq_length)
                event += tmp_event
                event_length += tmp_event_length
                label += tmp_label
                label_length += tmp_label_length
                del tmp_event
                del tmp_event_length
                del tmp_label
                del tmp_label_length
                count = len(event)
                if file_count % 10 == 0:
                    if max_segments_num is not None:
                        sys.stdout.write("%d/%d events read.   \n" % (
                        count, max_segments_num))
                        if len(event) > max_segments_num:
                            event.resize(max_segments_num)
                            label.resize(max_segments_num)
                            event_length.resize(max_segments_num)

                            label_length.resize(max_segments_num)
                            break
                    else:
                        sys.stdout.write("%d lines read.   \n" % (count))
                file_count += 1
    if event.cache:
        train = read_cache_dataset(h5py_file_path)
    else:
        train = DataSet(event=event, event_length=event_length, label=label,
                        label_length=label_length)
    return train


def read_signal(file_path, normalize="median"):
    f_h = open(file_path, 'r')
    signal = list()
    for line in f_h:
        signal += [float(x) for x in line.split()]
    signal = np.asarray(signal)
    if len(signal) == 0:
        return signal.tolist()
    if normalize == "mean":
        signal = (signal - np.mean(signal)) / np.float(np.std(signal))
    elif normalize == "median":
        signal = (signal - np.median(signal)) / np.float(robust.mad(signal))
    return signal.tolist()

def read_signal_tfrecord(data_array, normalize="median"):

    signal = data_array
    if len(signal) == 0:
        return signal.tolist()
    if normalize == "mean":
        signal = (signal - np.mean(signal)) / np.float(np.std(signal))
    elif normalize == "median":
        signal = (signal - np.median(signal)) / np.float(robust.mad(signal))
    return signal.tolist()


def read_label(file_path, skip_start=10, window_n=0):
    f_h = open(file_path, 'r')
    start = list()
    length = list()
    base = list()
    all_base = list()
    if skip_start < window_n:
        skip_start = window_n
    for line in f_h:
        print ('line', line)
        record = line.split()
        print ('record', record)
        exit()
        all_base.append(base2ind(record[2]))
    f_h.seek(0, 0)  # Back to the start
    file_len = len(all_base)
    for count, line in enumerate(f_h):
        record = line.split()
        if count < skip_start or count > (file_len - skip_start - 1):
            continue
        start.append(int(record[0]))
        length.append(int(record[1]) - int(record[0]))
        k_mer = 0
        for i in range(window_n * 2 + 1):
            k_mer = k_mer * 4 + all_base[count + i - window_n]
        base.append(k_mer)
    return raw_labels(start=start, length=length, base=base)


def read_label_tfrecord(raw_label_array, skip_start=10, window_n=0):
    start = list()
    length = list()
    base = list()
    all_base = list()
    count = 0
    window_n = int(window_n)
    if skip_start < window_n:
        skip_start = window_n
    for line in raw_label_array:
        all_base.append(base2ind(line[2]))
    file_len = len(all_base)
    for count, line in enumerate(raw_label_array):
        if count < skip_start or count > (file_len - skip_start - 1):
            continue
        start.append(int(line[0]))
        length.append(int(line[1]) - int(line[0]))
        k_mer = 0
        for i in range(window_n * 2 + 1):
            k_mer = k_mer * 4 + all_base[count + i - window_n]
        base.append(k_mer)
    return raw_labels(start=start, length=length, base=base)


def read_raw(raw_signal, raw_label, max_seq_length):
    label_val = list()
    label_length = list()
    event_val = list()
    event_length = list()
    current_length = 0
    current_label = []
    current_event = []
    for indx, segment_length in enumerate(raw_label.length):
        current_start = raw_label.start[indx]
        current_base = raw_label.base[indx]
        if current_length + segment_length < max_seq_length:
            current_event += raw_signal[
                             current_start:current_start + segment_length]
            current_label.append(current_base)
            current_length += segment_length
        else:
            # Save current event and label, conduct a quality controle step of the label.
            if current_length > (max_seq_length / 2) and len(current_label) > 5:
                padding(current_event, max_seq_length,
                        raw_signal[
                        current_start + segment_length:current_start + segment_length + max_seq_length])
                event_val.append(current_event)
                event_length.append(current_length)
                label_val.append(current_label)
                label_length.append(len(current_label))
                # Begin a new event-label
            current_event = raw_signal[
                            current_start:current_start + segment_length]
            current_length = segment_length
            current_label = [current_base]
    return event_val, event_length, label_val, label_length


def padding(x, L, padding_list=None):
    """Padding the vector x to length L"""
    len_x = len(x)
    assert len_x <= L, "Length of vector x is larger than the padding length"
    zero_n = L - len_x
    if padding_list is None:
        x.extend([0] * zero_n)
    elif len(padding_list) < zero_n:
        x.extend(padding_list + [0] * (zero_n - len(padding_list)))
    else:
        x.extend(padding_list[0:zero_n])
    return None


def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor
    """
    values = []
    indices = []
    for batch_i, label_list in enumerate(label_batch[:, 0]):
        for indx, label in enumerate(label_list):
            if indx >= label_batch[batch_i, 1]:
                break
            indices.append([batch_i, indx])
            values.append(label)
    shape = [len(label_batch), max(label_batch[:, 1])]
    return indices, values, shape


def base2ind(base, alphabet_n=4, base_n=1):
    """base to 1-hot vector,
    Input Args:
        base: current base,can be AGCT, or AGCTX for methylation.
        alphabet_n: can be 4 or 5, related to normal DNA or methylation call.
        """
    if alphabet_n == 4:
        Alphabeta = ['A', 'C', 'G', 'T']
        alphabeta = ['a', 'c', 'g', 't']
    elif alphabet_n == 5:
        Alphabeta = ['A', 'C', 'G', 'T', 'X']
        alphabeta = ['a', 'c', 'g', 't', 'x']
    else:
        raise ValueError('Alphabet number should be 4 or 5.')
    if base.isdigit():
        return int(base) / 256
    if ord(base) < 97:
        return Alphabeta.index(base)
    else:
        return alphabeta.index(base)
    #


def main():
    ### Input Test ###
    Data_dir = "/media/Linux_ex/Nanopore_Data/20170322_c4_watermanag_S10/tfrecord_test/"
    train = read_tfrecord(Data_dir,"train.tfrecords",seq_length=1000)
    for i in range(100):
        inputX, sequence_length, label = train.next_batch(10)


if __name__ == '__main__':
    main()

