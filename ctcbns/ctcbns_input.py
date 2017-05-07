#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:04:57 2017

@author: haotian.teng
"""
import numpy as np
import os,collections,sys

raw_labels = collections.namedtuple('raw_labels',['start','length','base'])


#FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_float(
#      'max_reads_number',
#      default_value = None,
#      docstring = "Max reads number input."
#                          )
class Flags():
    def __init__(self):
        self.max_reads_number = 500000
#        self.max_segment_len = 200
FLAGS = Flags()

        
class DataSet(object):
    def __init__(self,
                 event,
                 event_length,
                 label,
                 label_length,
                 for_eval = False,
                 ):
        """Custruct a DataSet."""
        if for_eval ==False:
            assert len(event)==len(label) and len(event_length)==len(label_length) and len(event)==len(event_length),"Sequence length for event \
            and label does not of event and label should be same, \
            event:%d , label:%d"%(len(event),len(label))
        self._event = np.asarray(zip(event,event_length))
        self._label = np.asarray(zip(label,label_length))
        self._reads_n = len(event)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._for_eval = for_eval
        
    @property
    def event(self):
        return self._event    
    @property
    def label(self):
        return self._label
    @property
    def reads_n(self):
        return self._reads_n
    @property 
    def index_in_epoch(self):
        return self._index_in_epoch
    @property 
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size,shuffle = True):
        """Return next batch in batch_size from the data set.
            Input Args:
                batch_size:batch size
                shuffle: boolean, indicate suffle or not
            Output Args:
                inputX,sequence_length,label_batch: tuple of (indx,vals,shape)"""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = np.arange(self._reads_n)
          np.random.shuffle(perm0)
          self._event = self.event[perm0]
          self._label = self.label[perm0]
        # Go to the next epoch
        if start + batch_size > self._reads_n:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest samples in this epoch
          rest_reads_n = self._reads_n - start
          event_rest_part = self._event[start:self._reads_n]
          label_rest_part = self._label[start:self._reads_n]
          # Shuffle the data
          if shuffle:
            perm = np.arange(self._reads_n)
            np.random.shuffle(perm)
            self._event = self.event[perm]
            self._label = self.label[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_reads_n
          end = self._index_in_epoch
          event_new_part = self._event[start:end]
          label_new_part = self._label[start:end]
          event_batch = np.concatenate((event_rest_part, event_new_part), axis=0)
          label_batch = np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          event_batch = self._event[start:end]
          label_batch = self._label[start:end]
        if not self._for_eval:
            label_batch = batch2sparse(label_batch)
        seq_length = event_batch[:,1].astype(np.int32)
        return np.vstack(event_batch[:,0]).astype(np.float32),seq_length,label_batch
def read_data_for_eval(file_path,start_index=0,step = 20,seg_length = 200):
    '''
    Input Args:
        file_path: file path to a signal file.
        start_index: the index of the signal start to read.
    '''
    if not file_path.endswith('.signal'):
        raise ValueError('A .signal file is required.')
    else:
        event = list()
        event_len = list()
        label = list()
        label_len = list()
        f_signal = read_signal(file_path,normalize = True)
        f_signal = f_signal[start_index:]
        sig_len = len(f_signal)
        for indx in range(0,sig_len,step):
            segment_sig = f_signal[indx:indx+seg_length]
            segment_len = len(segment_sig)
            event.append(padding(segment_sig,seg_length))
            event_len.append(segment_len)
        evaluation = DataSet(event = event,event_length = event_len,label = label,label_length = label_len,for_eval = True)
    return evaluation
def read_raw_data_sets(data_dir,seq_length = 200,k_mer = 1,valid_reads_num = 0):
    ###Read from raw data
    event = list()
    event_length = list()
    label = list()
    label_length = list()
    count = 0
    file_count = 0
    for name in os.listdir(data_dir):
        if name.endswith(".signal"):
            file_pre = os.path.splitext(name)[0]
            f_signal = read_signal(data_dir+name)
            try:
                f_label = read_label(data_dir+file_pre+'.label',skip_start = 10,window_n = (k_mer-1)/2)
            except:
                sys.stdout.write("Read the label %s fail.Skipped."%(name))
                continue

#            if seq_length<max(f_label.length):
#                print("Sequence length %d is samller than the max raw segment length %d, give a bigger seq_length"\
#                                 %(seq_length,max(f_label.length)))
#                l_indx = range(len(f_label.length))
#                for_sort = zip(l_indx,f_label.length)
#                sorted_array = sorted(for_sort,key = lambda x : x[1],reverse = True)
#                index = sorted_array[0][0]
#                plt.plot(f_signal[f_label.start[index]-100:f_label.start[index]+f_label.length[index]+100])
#                continue
            tmp_event,tmp_event_length,tmp_label,tmp_label_length = read_raw(f_signal,f_label,seq_length)
            event+=tmp_event
            event_length+=tmp_event_length
            label+=tmp_label
            label_length+=tmp_label_length
            count = len(event)
            if file_count%10 ==0:
                if FLAGS.max_reads_number is not None:
                    sys.stdout.write("%d/%d events read.   \n" % (count,FLAGS.max_reads_number) )
                    if len(event)>FLAGS.max_reads_number:
                        event = event[:FLAGS.max_reads_number]
                        label = label[:FLAGS.max_reads_number]
                        event_length = event_length[:FLAGS.max_reads_number]
                        label_length = label_length[:FLAGS.max_reads_number]
                        break
                else:
                    sys.stdout.write("%d lines read.   \n" % (count) )
            file_count+=1
#            print("Successfully read %d"%(file_count))
    assert valid_reads_num < len(event),"Valid reads number bigger than the total reads number."
    train_event = event[valid_reads_num:]
    train_event_length = event_length[valid_reads_num:]
    train_label = label[valid_reads_num:]
    train_label_length = label_length[valid_reads_num:]
    valid_event = event[:valid_reads_num]
    valid_event_length = event_length[:valid_reads_num]
    valid_label = label[:valid_reads_num]
    valid_label_length = label_length[:valid_reads_num]
    train = DataSet(event = train_event,event_length = train_event_length,label = train_label,label_length = train_label_length)
    valid = DataSet(event = valid_event,event_length = valid_event_length,label = valid_label,label_length = valid_label_length)
    return (train,valid)
def read_signal(file_path,normalize = True):
    f_h = open(file_path,'r')
    signal = list()
    for line in f_h:
        signal+=[int(x) for x in line.split()]
    signal = np.asarray(signal)
    if normalize:
        signal = (signal - np.mean(signal))/np.std(signal)
    return signal.tolist()

def read_label(file_path,skip_start=10,window_n = 0):
    f_h = open(file_path,'r')
    start = list()
    length = list()
    base = list()
    all_base = list()
    count = 0
    if skip_start < window_n:
        skip_start = window_n
    for line in f_h:
        record = line.split()
        all_base.append(base2ind(record[2]))
    f_h.seek(0,0)#Back to the start
    file_len = len(all_base)
    for count,line in enumerate(f_h):
        record = line.split()
        if count<skip_start or count>(file_len-skip_start-1):
            continue
        start.append(int(record[0]))
        length.append(int(record[1])-int(record[0]))
        k_mer = 0
        for i in range(window_n*2+1):
            k_mer = k_mer*4 + all_base[count+i-window_n]
        base.append(k_mer)
    return raw_labels(start=start,length=length,base=base)
def read_raw(raw_signal,raw_label,max_seq_length):
    label_val = list()
    label_length=list()
    event_val = list()
    event_length = list()
    current_length = 0
    current_label = []
    current_event = []
    for indx,segment_length in enumerate(raw_label.length):
        current_start = raw_label.start[indx]
        current_base = raw_label.base[indx]
        if current_length+segment_length<max_seq_length:
            current_event += raw_signal[current_start:current_start+segment_length]
            current_label.append(current_base)
            current_length+= segment_length
        else:
            #Save current event and label
            if current_length>(max_seq_length/2) and len(current_label)>5:
                current_event = padding(current_event,max_seq_length,raw_signal[current_start+segment_length:current_start+segment_length+max_seq_length])
                event_val.append(current_event)
                event_length.append(current_length)
                label_val.append(current_label)
                label_length.append(len(current_label)) 
            #Begin a new event-label
            current_event = raw_signal[current_start:current_start+segment_length]
            current_length = segment_length
            current_label = [current_base]
    return event_val,event_length,label_val,label_length
            
def padding(x,L,padding_list = None):
    """Padding the vector x to length L"""
    len_x = len(x)
    assert len_x<=L, "Length of vector x is larger than the padding length"
    zero_n = L-len_x
    if padding_list is None:
        return x+[0]*zero_n
    elif len(padding_list)<zero_n:
        return x+padding_list+[0]*(zero_n-len(padding_list))
    else:
        return x+padding_list[0:zero_n]
def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor"""    
    values = []
    indices = []
    for batch_i,label_list in enumerate(label_batch[:,0]):
        for indx,label in enumerate(label_list):
            indices.append([batch_i,indx])
            values.append(label)
    shape = [len(label_batch),max(label_batch[:,1])]
    return (indices,values,shape)
def base2ind(base,alphabet_n = 4,base_n = 1):
    """base to 1-hot vector,
    Input Args:
        base: current base,can be AGCT, or AGCTX for methylation.
        alphabet_n: can be 4 or 5, related to normal DNA or methylation call.
        """
    if alphabet_n == 4:
        Alphabeta = ['A','C','G','T']
        alphabeta = ['a','c','g','t']
    elif alphabet_n==5:
        Alphabeta = ['A','C','G','T','X']
        alphabeta = ['a','c','g','t','x']
    else:
        raise ValueError('Alphabet number should be 4 or 5.')
    if base.isdigit():
        return int(base)/256
    if ord(base)<97:
        return Alphabeta.index(base)
    else:
        return alphabeta.index(base)                   

def main():
    Data_dir = "/home/haotianteng/UQ/deepBNS/data/test/raw/"
    train,valid = read_raw_data_sets(Data_dir,seq_length = 1000)
    for i in range(100):
        inputX,sequence_length,label = train.next_batch(10)
        indxs,values,shape = label
if __name__=='__main__':
    main()
    
     
            
