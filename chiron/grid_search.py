#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:36:04 2019

@author: heavens
"""
import json
import itertools

configure = {}
#Setting the static configuration
configure["cnn"] = {"model":"dynamic_net"}
configure["rnn"] = {"layer_num":3,
                    "cell_type":"LSTM"}
configure["opt_method"] = "Momentum"
configure["fl_gamma"]  = 2
#Saving grid configuration:
cnn_layer_grid = [['res']*3]
hidden_num_grid = [[128]*3,[256]*3,[512]*3]
kernals_grid = [[5,5,5],[15,3,3],[15,3,3]]
strides_grid = [[2,2,2],[7,1,1],[5,1,1]]
rnn_hu_grid = [100,200]
itretools.product([0],[0,1,2],[0,1,2],[0,1])

###TODO### Finish the script to generate multiple model configuration json files.
