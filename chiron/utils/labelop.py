# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import absolute_import
from __future__ import print_function
import h5py
import numpy as np
from six.moves import zip


def get_label_segment(fast5_fn, basecall_group, basecall_subgroup):
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get samping rate
    try:
        fast5_info = fast5_data['UniqueGlobalKey/channel_id'].attrs
        sampling_rate = fast5_info['sampling_rate'].astype('int_')
    except:
        raise RuntimeError(('Could not get channel info'))

    # Read raw data
    try:
        raw_dat = list(fast5_data['/Raw/Reads/'].values())[0]
        raw_attrs = raw_dat.attrs
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')
    raw_start_time = raw_attrs['start_time']

    # Read segmented data
    try:
        segment_dat = fast5_data[
            '/Analyses/' + basecall_group + '/' + basecall_subgroup + '/Events']
        segment_attrs = dict(list(segment_dat.attrs.items()))
        segment_dat = segment_dat.value

        total = len(segment_dat)

        # Process segment data
        segment_starts = segment_dat['start'] * sampling_rate - raw_start_time
        segment_lengths = segment_dat['length'] * sampling_rate
        segment_means = segment_dat['mean']
        segment_stdv = segment_dat['stdv']

        # create the label for segment event
        segment_kmer = np.full(segment_starts.shape, '-', dtype='S5')
        segment_move = np.zeros(segment_starts.shape)
        segment_cstart = np.zeros(segment_starts.shape)
        segment_clength = np.zeros(segment_starts.shape)

        segment_data = np.array(
            list(zip(segment_means, segment_stdv, segment_starts,
                     segment_lengths, segment_kmer, segment_move,
                     segment_cstart, segment_clength)),
            dtype=[('mean', 'float64'), ('stdv', 'float64'), ('start', '<u4'),
                   ('length', '<u4'), ('kmer', 'S5'),
                   ('move', '<u4'), ('cstart', '<u4'), ('clength', '<u4')])

    except:
        raise RuntimeError(
            'No events or corrupted events in file. Likely a ' +
            'segmentation error or mis-specified basecall-' +
            'subgroups (--2d?).')
    try:
        # Read corrected data
        corr_dat = fast5_data[
            '/Analyses/RawGenomeCorrected_000/' + basecall_subgroup + '/Events']
        corr_attrs = dict(list(corr_dat.attrs.items()))
        corr_dat = corr_dat.value
    except:
        raise RuntimeError((
            'Corrected data now found.'))

    corr_start_time = corr_attrs['read_start_rel_to_raw']
    corr_starts = corr_dat['start'] + corr_start_time
    corr_lengths = corr_dat['length']
    corr_bases = corr_dat['base']

    fast5_data.close()

    first_segment_index = 0
    corr_index = 2
    kmer = ''.join(corr_bases[0:5])

    # Move segment to the first available corr_data
    while segment_data[first_segment_index]['start'] < corr_starts[corr_index]:
        first_segment_index += 1

    segment_index = first_segment_index
    move = 0
    while segment_index < len(segment_data):
        my_start = corr_starts[corr_index]
        my_length = corr_lengths[corr_index]
        my_end = my_start + corr_lengths[corr_index]
        move += 1
        while True:
            segment_data[segment_index]['kmer'] = kmer
            segment_data[segment_index]['cstart'] = my_start
            segment_data[segment_index]['clength'] = my_length
            segment_data[segment_index]['move'] = move
            segment_data[segment_index]['kmer'] = kmer
            move = 0

            # if segment_data[segment_index]['start'] + segment_data[segment_index]['length'] < my_end:
            #    move = 0
            segment_index += 1
            if (segment_index >= len(segment_data)):
                break

            if segment_data[segment_index]['start'] >= my_end:
                break
            # End of while true
        corr_index += 1

        if corr_index >= len(corr_starts) - 2:
            break
        kmer = kmer[1:] + corr_bases[corr_index + 2]

    #    print first_segment_index
    #    print segment_index
    #    print corr_index
    segment_data = segment_data[first_segment_index:segment_index]
    return (segment_data, first_segment_index, segment_index, total)


def get_label_raw(fast5_fn, basecall_group, basecall_subgroup,reverse = False):
    ##Open file
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:
        raw_dat = list(fast5_data['/Raw/Reads/'].values())[0]
        # raw_attrs = raw_dat.attrs
        raw_dat = raw_dat['Signal'].value
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')

    # Read corrected data
    try:
        corr_data = fast5_data[
            '/Analyses/'+basecall_group +'/' + basecall_subgroup + '/Events']
        corr_attrs = dict(list(corr_data.attrs.items()))
        corr_data = corr_data.value
    except:
        raise RuntimeError((
            'Corrected data not found.'))

    fast5_info = fast5_data['UniqueGlobalKey/channel_id'].attrs
    # sampling_rate = fast5_info['sampling_rate'].astype('int_')

    # Reading extra information
    corr_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']  #
    if len(raw_dat) > 99999999:
        raise ValueError(fast5_fn + ": max signal length exceed 99999999")
    if any(len(vals) <= 1 for vals in (
            corr_data, raw_dat)):
        raise NotImplementedError((
            'One or no segments or signal present in read.'))
    event_starts = corr_data['start'] + corr_start_rel_to_raw
    event_lengths = corr_data['length']
    event_bases = corr_data['base']

    fast5_data.close()
    label_data = np.array(
        list(zip(event_starts, event_lengths, event_bases)),
        dtype=[('start', '<u4'), ('length', '<u4'), ('base', 'S1')])
    return (raw_dat, label_data, event_starts, event_lengths)


def write_label_segment(fast5_fn, raw_label, segment_label, first, last):
    fast5_data = h5py.File(fast5_fn, 'r+')
    analyses_grp = fast5_data['/Analyses']

    label_group = "LabeledData"
    if label_group in analyses_grp:
        del analyses_grp[label_group]

    label_grp = analyses_grp.create_group(label_group)
    label_subgroup = label_grp.create_group(basecall_subgroup)

    label_subgroup.create_dataset(
        'raw_data', data=raw_data, compression="gzip")

    raw_label_data = label_subgroup.create_dataset(
        'raw_label', data=raw_label, compression="gzip")

    segment_label_data = label_subgroup.create_dataset(
        'segment_label', data=segment_label, compression="gzip")

    segment_label_data.attrs['first'] = first
    segment_label_data.attrs['last'] = last

    fast5_data.flush()
    fast5_data.close()


if __name__ == '__main__':
    fast5_fn = "/home/haotianteng/UQ/deepBNS/data/test/pass/test.fast5"

    basecall_subgroup = 'BaseCalled_template'
    basecall_group = 'RawGenomeCorrected_000'

    # Get segment data
    (segment_label, first_segment, last_segment, total) = get_label_segment(
        fast5_fn, basecall_group, basecall_subgroup)

    # segment_label is the numpy array containing labeling of the segment
    print((
        "There are {} segments, and {} are labeled ({},{})".format(total,
                                                                   last_segment - first_segment,
                                                                   first_segment,
                                                                   last_segment)))

    # get raw data
    (raw_data, raw_label, raw_start, raw_length) = get_label_raw(fast5_fn,
                                                                 basecall_group,
                                                                 basecall_subgroup)

    # You can write the labels back to the fast5 file for easy viewing with hdfviewer
    # write_label_segment(fast5_fn, raw_label, segment_label, first, last)
