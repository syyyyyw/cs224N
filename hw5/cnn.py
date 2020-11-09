#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    '''
        input_channel char_embedded
        output_channel word_embedded

    '''
    def __init__(self,input_channel,output_channel,kernel_size=5,padding=1):
        super(CNN,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv1d(self.input_channel,self.output_channel,self.kernel_size,padding=self.padding)
    def forward(self,x):
        X_conv=self.conv(x)
        max_word_len=x.shape[-1]
        max_pool=nn.MaxPool1d(max_word_len-self.kernel_size+2*self.padding+1)
        X_conv_out=max_pool(F.relu(X_conv)).squeeze(-1)

        return X_conv_out


    ### END YOUR CODE

