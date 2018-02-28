#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:20:45 2017
Read File batch test
@author: haotianteng
"""
import struct

file_path = '/media/haotianteng/Linux_ex/Nanopore_Data/Lambda_R9.4/file_batch/data_batch_1.bin'
with open(file_path, mode='rb') as file:
    fileContent = file.read()
print(struct.unpack("<1H512f1H512b", fileContent[:2564]))
