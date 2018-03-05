# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Sun Dec 24 15:20:45 2017
#Read File batch test

import struct

file_path = '/media/haotianteng/Linux_ex/Nanopore_Data/Lambda_R9.4/file_batch/data_batch_1.bin'
with open(file_path, mode='rb') as file:
    fileContent = file.read()
print(struct.unpack("<1H512f1H512b", fileContent[:2564]))
