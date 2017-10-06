#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:04:57 2017

@author: haotian.teng
"""
import numpy as np
import os,collections,sys
import h5py
import tempfile
raw_labels = collections.namedtuple('raw_labels',['start','length','base'])

class Flags(object):
    def __init__(self):
        self.max_reads_number = 100000
        self.MAXLEN = 1e5 #Maximum Length of the holder in biglist. 1e6 by default
#        self.max_segment_len = 200
FLAGS = Flags()

class biglist(object):
    #Read into memory if reads number < MAXLEN, otherwise read into h5py database
    def __init__(self,data_handle,dtype = 'float32'):
        self.handle = data_handle
        self.dtype = dtype
        self.holder = list()
        self.len = 0
        self.cache = False #Mark if the list has been saved into hdf5 or not
    @property
    def shape(self):
        return self.handle.shape
    def append(self,item):
        self.holder.append(item)
        self.check_save
    def __add__(self,add_list):
        self.holder += add_list
        self.check_save()
        return self
    def __len__(self):
        return self.len + len(self.holder)
    def resize(self,size,axis = 0):
        self.save_rest()
        if self.cache:
            self.handle.resize(size,axis = axis  )
            self.len = len(self.handle)
        else:
            self.holder = self.holder[:size]
    def save_rest(self):
        if self.cache:
         if len(self.holder)!=0:
            self.save()
    def check_save(self):
        if len(self.holder) > FLAGS.MAXLEN:
         self.save()
         self.cache = True
        
    def save(self):
        if type(self.holder[0]) is list:
            max_sub_len = max([len(sub_a) for sub_a in self.holder])
            shape = self.handle.shape
            for item in self.holder:
                item.extend([0]*(max(shape[1],max_sub_len) - len(item)))
            if max_sub_len > shape[1]:
                self.handle.resize(max_sub_len,axis = 1)
            self.handle.resize(self.len+len(self.holder),axis = 0)
            self.handle[self.len:] = self.holder
            self.len+=len(self.holder)
            del self.holder[:]
            self.holder = list()
        else:
            self.handle.resize(self.len+len(self.holder),axis = 0)
            self.handle[self.len:] = self.holder
            self.len+=len(self.holder)
            del self.holder[:]
            self.holder = list()
    def __getitem__(self,val):
        if self.cache:
         if len(self.holder)!=0:
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
                 for_eval = False,
                 ):
        """Custruct a DataSet."""
        if for_eval ==False:
            assert len(event)==len(label) and len(event_length)==len(label_length) and len(event)==len(event_length),"Sequence length for event \
            and label does not of event and label should be same, \
            event:%d , label:%d"%(len(event),len(label))
        self._event = event
        self._event_length = event_length
        self._label = label
        self._label_length=label_length
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
    
    def read_into_memory(self,index):
        event = np.asarray(zip([self._event[i] for i in index],[self._event_length[i] for i in index]))
        label = np.asarray(zip([self._label[i] for i in index],[self._label_length[i] for i in index]))
        return event,label
    def next_batch(self, batch_size,shuffle = True):
        """Return next batch in batch_size from the data set.
            Input Args:
                batch_size:batch size
                shuffle: boolean, indicate suffle or not
            Output Args:
                inputX,sequence_length,label_batch: tuple of (indx,vals,shape)"""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
         if shuffle:
          np.random.shuffle(self._perm)
        # Go to the next epoch
        if start + batch_size > self._reads_n:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest samples in this epoch
          rest_reads_n = self._reads_n - start
          event_rest_part,label_rest_part = self.read_into_memory(self._perm[start:self._reads_n])
           
          # Shuffle the data
          if shuffle:
            np.random.shuffle(self._perm)
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_reads_n
          end = self._index_in_epoch
          event_new_part,label_new_part = self.read_into_memory(self._perm[start:end])
          
          event_batch = np.concatenate((event_rest_part, event_new_part), axis=0)
          label_batch = np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          event_batch,label_batch = self.read_into_memory(self._perm[start:end])
          
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
            padding(segment_sig,seg_length)
            event.append(segment_sig)
            event_len.append(segment_len)
        evaluation = DataSet(event = event,event_length = event_len,label = label,label_length = label_len,for_eval = True)
    return evaluation
def read_raw_data_sets(data_dir,h5py_file_path=None,seq_length = 300,k_mer = 1,max_reads_num = FLAGS.max_reads_number):
    ###Read from raw data
    if h5py_file_path is None:
        h5py_file_path = tempfile.mkdtemp()+'/temp_record.hdf5'
    hdf5_record = h5py.File(h5py_file_path,"w")
    event_h = hdf5_record.create_dataset('event/record',dtype = 'float32', shape=(0,seq_length),maxshape = (None,seq_length))
    event_length_h = hdf5_record.create_dataset('event/length',dtype = 'int32',shape=(0,),maxshape =(None,),chunks =True )
    label_h = hdf5_record.create_dataset('label/record',dtype = 'int32',shape = (0,0),maxshape = (None,seq_length))
    label_length_h = hdf5_record.create_dataset('label/length',dtype = 'int32',shape = (0,),maxshape = (None,))
    event = biglist(data_handle = event_h)
    event_length = biglist(data_handle = event_length_h)
    label = biglist(data_handle = label_h)
    label_length = biglist(data_handle = label_length_h)
    count = 0
    file_count = 0
    for name in os.listdir(data_dir):
        if name.endswith(".signal"):
            file_pre = os.path.splitext(name)[0]
            f_signal = read_signal(data_dir+name)
            if len(f_signal)==0:
                continue
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
#                continueholder_
            tmp_event,tmp_event_length,tmp_label,tmp_label_length = read_raw(f_signal,f_label,seq_length)
            event+=tmp_event
            event_length+=tmp_event_length
            label+=tmp_label
            label_length+=tmp_label_length
            del tmp_event
            del tmp_event_length
            del tmp_label
            del tmp_label_length
            count = len(event)
            if file_count%10 ==0:
                if FLAGS.max_reads_number is not None:
                    sys.stdout.write("%d/%d events read.   \n" % (count,FLAGS.max_reads_number) )
                    if len(event)>FLAGS.max_reads_number:
                        event.resize(FLAGS.max_reads_number)
                        label.resize(FLAGS.max_reads_number)
                        event_length.resize(FLAGS.max_reads_number)
                        
                        label_length.resize(FLAGS.max_reads_number)
                        break
                else:
                    sys.stdout.write("%d lines read.   \n" % (count) )
            file_count+=1
#            print("Successfully read %d"%(file_count))
    train_event = event
    train_event_length = event_length
    train_label = label
    train_label_length = label_length
    train = DataSet(event = train_event,event_length = train_event_length,label = train_label,label_length = train_label_length)
    return train
def read_signal(file_path,normalize = True):
    f_h = open(file_path,'r')
    signal = list()
    for line in f_h:
        signal+=[float(x) for x in line.split()]
    signal = np.asarray(signal)
    if len(signal)==0:
        return signal.tolist()
    if normalize:
        signal = (signal - np.mean(signal))/np.float(np.std(signal))
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
            #Save current event and label, conduct a quality controle step of the label.
            if current_length>(max_seq_length/2) and len(current_label)>5:
                padding(current_event,max_seq_length,raw_signal[current_start+segment_length:current_start+segment_length+max_seq_length])
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
        x.extend([0]*zero_n)
    elif len(padding_list)<zero_n:
        x.extend(padding_list+[0]*(zero_n-len(padding_list)))
    else:
        x.extend(padding_list[0:zero_n])
    return None
def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor"""    
    values = []
    indices = []
    for batch_i,label_list in enumerate(label_batch[:,0]):
        for indx,label in enumerate(label_list):
            if indx>=label_batch[batch_i,1]:
                break
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
#
def main():
### Input Test ###
	Data_dir = "/media/haotianteng/Linux_ex/Nanopore_data/Lambda_R9.4/raw/"
	train = read_raw_data_sets(Data_dir,seq_length = 400)
	for i in range(100):
	    inputX,sequence_length,label = train.next_batch(10)
	    indxs,values,shape = label
if __name__=='__main__':
    main()
#    
#     
#            
#hdf5_record = h5py.File('/home/haotianteng/Documents/123/test2.hdf5',"w")
#event_h = hdf5_record.create_dataset('test2',dtype = 'float32', shape=(0,300),maxshape = (None,300))
